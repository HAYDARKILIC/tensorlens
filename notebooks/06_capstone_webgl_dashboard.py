"""Week 6 capstone — WebGL-based live LLM training dashboard.

A self-contained asynchronous monitoring server that streams training-time
telemetry (loss, gradient norm, VRAM, anisotropy index, top Hessian
eigenvalue, perplexity, learning rate) over a WebSocket to a browser
front-end that renders the data with Plotly.js WebGL traces.

Why WebGL?
----------
Standard Plotly/Matplotlib charts re-render the full canvas on every update
which becomes the bottleneck above ~5 Hz update rates with thousands of
points. Plotly's ``scattergl`` trace uses the browser's GPU, achieving
60 Hz on a million points. The wire protocol uses MessagePack for compact
binary encoding (~30% smaller than JSON, no string parsing) — the
"zero-copy" intent of the design.

Architecture
------------
*  An async **producer** task simulates / consumes training metrics and
   pushes them into a queue.
*  A single-page HTML+JS **dashboard** subscribes via WebSocket.
*  Both run inside one ``aiohttp`` application; no external dependencies
   beyond ``aiohttp``, ``msgpack``, and ``numpy``.

Run
---
::

    python notebooks/06_capstone_webgl_dashboard.py --port 8765

Then open ``http://localhost:8765/`` in any modern browser.

In production, swap the simulated producer for a thin training-loop hook
that publishes the same dict shape into ``MetricQueue.put``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import msgpack
import numpy as np
from aiohttp import WSMsgType, web

# Make the parent ``src`` package importable when running from the
# notebooks/ directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import get_logger  # noqa: E402

_log = get_logger("tensorlens.dashboard")


# ----------------------------------------------------------------------------
# Metric model
# ----------------------------------------------------------------------------


@dataclass
class TrainingMetric:
    """One snapshot of training telemetry.

    Attributes
    ----------
    step
        Optimizer step.
    timestamp
        Unix timestamp (seconds).
    loss
        Scalar loss.
    perplexity
        ``exp(loss)`` for autoregressive LM; otherwise NaN.
    grad_norm
        L2 norm of the flattened gradient.
    learning_rate
        Current scheduler LR.
    vram_gb
        Reserved VRAM in gigabytes.
    gpu_temp_c
        GPU temperature in Celsius.
    anisotropy
        Mean pairwise cosine of the last batch's hidden states.
    lambda_max
        Top Hessian eigenvalue (estimated).
    layer_grad_norms
        Per-layer gradient norms (length L).
    """

    step: int
    timestamp: float
    loss: float
    perplexity: float
    grad_norm: float
    learning_rate: float
    vram_gb: float
    gpu_temp_c: float
    anisotropy: float
    lambda_max: float
    layer_grad_norms: list[float] = field(default_factory=list)

    def to_msgpack(self) -> bytes:
        return msgpack.packb(asdict(self), use_bin_type=True)


# ----------------------------------------------------------------------------
# Simulated metric producer
# ----------------------------------------------------------------------------


class SimulatedTrainer:
    """Asynchronous metric generator emulating a real LLM trainer.

    Parameters
    ----------
    n_layers
        Number of layers reported in ``layer_grad_norms``.
    initial_lr
        Starting LR.
    rate_hz
        Update frequency in Hz.
    """

    def __init__(self, n_layers: int = 12, initial_lr: float = 3e-4, rate_hz: float = 5.0) -> None:
        self.n_layers = n_layers
        self.lr = initial_lr
        self.step = 0
        self.period_s = 1.0 / max(rate_hz, 1e-3)
        self._loss = 6.0
        self._aniso = 0.15
        self._lam = 12.0
        self._rng = np.random.default_rng(0)

    def _next(self) -> TrainingMetric:
        # Smooth descent + heteroscedastic noise
        self._loss = max(0.5, self._loss - 0.005 + 0.02 * self._rng.standard_normal())
        self._aniso = min(0.85, max(0.05, self._aniso + 0.002 + 0.005 * self._rng.standard_normal()))
        self._lam = max(1.0, self._lam + 0.05 * self._rng.standard_normal())
        # Cosine-decay LR schedule with warmup
        if self.step < 200:
            self.lr = (self.step + 1) / 200.0 * 3e-4
        else:
            self.lr = 3e-4 * 0.5 * (1.0 + math.cos(min(1.0, (self.step - 200) / 1000.0) * math.pi))
        grad_norm = max(0.05, 2.0 * abs(self._rng.standard_normal()) * math.exp(-self.step / 500))
        layer_norms = (grad_norm * (1.0 + 0.5 * self._rng.standard_normal(size=self.n_layers))).clip(0).tolist()
        self.step += 1
        return TrainingMetric(
            step=self.step,
            timestamp=time.time(),
            loss=float(self._loss),
            perplexity=float(math.exp(self._loss)),
            grad_norm=float(grad_norm),
            learning_rate=float(self.lr),
            vram_gb=72.0 + 0.5 * float(self._rng.standard_normal()),
            gpu_temp_c=65.0 + 5.0 * float(math.sin(self.step / 30.0)),
            anisotropy=float(self._aniso),
            lambda_max=float(self._lam),
            layer_grad_norms=[float(x) for x in layer_norms],
        )

    async def run(self, queue: "MetricQueue") -> None:
        """Loop forever, pushing new metrics into the queue."""
        try:
            while True:
                metric = self._next()
                await queue.put(metric)
                await asyncio.sleep(self.period_s)
        except asyncio.CancelledError:
            _log.info("SimulatedTrainer cancelled at step %d", self.step)
            raise


# ----------------------------------------------------------------------------
# Pub-sub queue
# ----------------------------------------------------------------------------


class MetricQueue:
    """Fan-out queue: one producer, many WebSocket subscribers.

    Each subscriber gets its own ``asyncio.Queue`` so a slow client cannot
    block a fast one. New subscribers join at the current point in the stream;
    historical metrics are not replayed.
    """

    def __init__(self, max_per_subscriber: int = 1024) -> None:
        self._subscribers: list[asyncio.Queue[TrainingMetric]] = []
        self._max = max_per_subscriber

    def subscribe(self) -> asyncio.Queue[TrainingMetric]:
        q: asyncio.Queue[TrainingMetric] = asyncio.Queue(maxsize=self._max)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[TrainingMetric]) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    async def put(self, metric: TrainingMetric) -> None:
        for q in list(self._subscribers):
            if q.full():
                # Drop oldest if a subscriber lags — bounded memory is essential.
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            await q.put(metric)


# ----------------------------------------------------------------------------
# WebSocket handler
# ----------------------------------------------------------------------------


async def _ws_handler(request: web.Request) -> web.WebSocketResponse:
    """Serve a WebSocket subscription to the metric stream."""
    queue: MetricQueue = request.app["queue"]
    ws = web.WebSocketResponse(heartbeat=20.0, max_msg_size=2**20)
    await ws.prepare(request)
    _log.info("client connected from %s", request.remote)
    sub = queue.subscribe()
    send_task: asyncio.Task[None] | None = None
    try:
        send_task = asyncio.create_task(_pump(ws, sub))
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                # Currently ignored; placeholder for future control commands.
                _log.debug("client text: %s", msg.data[:120])
            elif msg.type == WSMsgType.ERROR:
                _log.warning("ws error: %s", ws.exception())
    finally:
        if send_task is not None:
            send_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass
        queue.unsubscribe(sub)
        _log.info("client disconnected: %s", request.remote)
    return ws


async def _pump(ws: web.WebSocketResponse, sub: asyncio.Queue[TrainingMetric]) -> None:
    """Forward queued metrics to the WebSocket as MessagePack frames."""
    while True:
        metric = await sub.get()
        try:
            await ws.send_bytes(metric.to_msgpack())
        except ConnectionResetError:
            return


# ----------------------------------------------------------------------------
# Static dashboard HTML (Plotly.js + WebGL)
# ----------------------------------------------------------------------------


_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>TensorLens — Live Training Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@msgpack/msgpack@2.8.0/dist.umd/msgpack.min.js"></script>
  <style>
    :root { --bg:#0a0e17; --panel:#141a25; --fg:#e6edf3; --muted:#7d8590; --accent:#58a6ff; }
    html, body { background:var(--bg); color:var(--fg); margin:0; padding:0;
                 font-family:'IBM Plex Mono', monospace; height:100%; }
    header { padding:14px 24px; border-bottom:1px solid #21262d; display:flex;
             justify-content:space-between; align-items:center; }
    h1 { margin:0; font-size:18px; letter-spacing:0.5px; }
    .status { color:var(--muted); font-size:13px; }
    .grid { display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; padding:12px; }
    .card { background:var(--panel); border:1px solid #21262d; border-radius:6px;
            padding:10px; min-height:240px; }
    .stat { font-size:11px; color:var(--muted); text-transform:uppercase;
            letter-spacing:1px; }
    .value { font-size:28px; color:var(--accent); }
    footer { padding:8px 24px; color:var(--muted); font-size:11px; }
  </style>
</head>
<body>
  <header>
    <h1>TENSORLENS — live training telemetry</h1>
    <div class="status" id="status">connecting…</div>
  </header>
  <section class="grid">
    <div class="card">
      <div class="stat">loss / step</div><div class="value" id="loss-val">—</div>
      <div id="loss-plot" style="height:170px"></div>
    </div>
    <div class="card">
      <div class="stat">perplexity</div><div class="value" id="ppl-val">—</div>
      <div id="ppl-plot" style="height:170px"></div>
    </div>
    <div class="card">
      <div class="stat">gradient norm</div><div class="value" id="grad-val">—</div>
      <div id="grad-plot" style="height:170px"></div>
    </div>
    <div class="card">
      <div class="stat">anisotropy 𝒜</div><div class="value" id="aniso-val">—</div>
      <div id="aniso-plot" style="height:170px"></div>
    </div>
    <div class="card">
      <div class="stat">Hessian λmax</div><div class="value" id="lam-val">—</div>
      <div id="lam-plot" style="height:170px"></div>
    </div>
    <div class="card">
      <div class="stat">VRAM (GB) · GPU temp (°C)</div>
      <div class="value" id="vram-val">—</div>
      <div id="vram-plot" style="height:170px"></div>
    </div>
    <div class="card" style="grid-column:1 / span 3">
      <div class="stat">per-layer gradient norms (live heatmap)</div>
      <div id="layer-plot" style="height:280px"></div>
    </div>
  </section>
  <footer>WebSocket • MessagePack • Plotly.js WebGL</footer>

<script>
const WINDOW = 600;
const buf = { step:[], loss:[], ppl:[], grad:[], aniso:[], lam:[], vram:[], temp:[] };
const layerHistory = []; // 2D: rows = time, cols = layers
const PLOT_OPTS = { displayModeBar:false, responsive:true };
const LINE_TPL = (color) => ({ mode:'lines', type:'scattergl',
    line:{color, width:1.5}, hoverinfo:'skip' });
const LAYOUT_TPL = (color) => ({
    margin:{t:4,b:24,l:36,r:6},
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
    xaxis:{color:'#7d8590', gridcolor:'#21262d', zeroline:false},
    yaxis:{color:'#7d8590', gridcolor:'#21262d', zeroline:false},
    font:{color:'#e6edf3', family:'IBM Plex Mono'}
});

function initCharts() {
  Plotly.newPlot('loss-plot',  [{ x:[], y:[], ...LINE_TPL('#58a6ff') }], LAYOUT_TPL(), PLOT_OPTS);
  Plotly.newPlot('ppl-plot',   [{ x:[], y:[], ...LINE_TPL('#a5d6ff') }], LAYOUT_TPL(), PLOT_OPTS);
  Plotly.newPlot('grad-plot',  [{ x:[], y:[], ...LINE_TPL('#ffa657') }], LAYOUT_TPL(), PLOT_OPTS);
  Plotly.newPlot('aniso-plot', [{ x:[], y:[], ...LINE_TPL('#7ee787') }], LAYOUT_TPL(), PLOT_OPTS);
  Plotly.newPlot('lam-plot',   [{ x:[], y:[], ...LINE_TPL('#f778ba') }], LAYOUT_TPL(), PLOT_OPTS);
  Plotly.newPlot('vram-plot',  [
      { x:[], y:[], ...LINE_TPL('#79c0ff'), name:'vram' },
      { x:[], y:[], ...LINE_TPL('#ffa657'), name:'temp', yaxis:'y2' }],
      { ...LAYOUT_TPL(), yaxis2:{overlaying:'y', side:'right', color:'#ffa657',
                                 gridcolor:'rgba(0,0,0,0)'} },
      PLOT_OPTS);
  Plotly.newPlot('layer-plot', [{ z:[], type:'heatmap', colorscale:'Viridis',
                                   colorbar:{thickness:8, len:0.7, x:1.02, tickfont:{color:'#7d8590'}}}],
      LAYOUT_TPL(), PLOT_OPTS);
}
initCharts();

function push(buf, key, x, y, win) {
  buf[key].push(y);
  if (buf[key].length > win) buf[key].shift();
}

function applyUpdate(m) {
  const x = m.step;
  push(buf, 'step', x, x, WINDOW);
  push(buf, 'loss', x, m.loss, WINDOW);
  push(buf, 'ppl', x, m.perplexity, WINDOW);
  push(buf, 'grad', x, m.grad_norm, WINDOW);
  push(buf, 'aniso', x, m.anisotropy, WINDOW);
  push(buf, 'lam', x, m.lambda_max, WINDOW);
  push(buf, 'vram', x, m.vram_gb, WINDOW);
  push(buf, 'temp', x, m.gpu_temp_c, WINDOW);

  document.getElementById('loss-val').textContent  = m.loss.toFixed(3);
  document.getElementById('ppl-val').textContent   = m.perplexity.toFixed(2);
  document.getElementById('grad-val').textContent  = m.grad_norm.toFixed(3);
  document.getElementById('aniso-val').textContent = m.anisotropy.toFixed(3);
  document.getElementById('lam-val').textContent   = m.lambda_max.toFixed(2);
  document.getElementById('vram-val').textContent  =
      m.vram_gb.toFixed(1) + ' / ' + m.gpu_temp_c.toFixed(1) + '°';

  Plotly.update('loss-plot',  { x:[buf.step], y:[buf.loss] },  {}, [0]);
  Plotly.update('ppl-plot',   { x:[buf.step], y:[buf.ppl] },   {}, [0]);
  Plotly.update('grad-plot',  { x:[buf.step], y:[buf.grad] },  {}, [0]);
  Plotly.update('aniso-plot', { x:[buf.step], y:[buf.aniso] }, {}, [0]);
  Plotly.update('lam-plot',   { x:[buf.step], y:[buf.lam] },   {}, [0]);
  Plotly.update('vram-plot',  { x:[buf.step, buf.step], y:[buf.vram, buf.temp] }, {}, [0,1]);

  layerHistory.push(m.layer_grad_norms);
  if (layerHistory.length > WINDOW) layerHistory.shift();
  Plotly.update('layer-plot', { z:[transpose(layerHistory)] }, {}, [0]);
}

function transpose(m) {
  if (m.length === 0) return [];
  const rows = m.length, cols = m[0].length;
  const out = Array.from({length:cols}, () => new Array(rows));
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++)
      out[j][i] = m[i][j];
  return out;
}

const url = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws';
const ws = new WebSocket(url);
ws.binaryType = 'arraybuffer';
ws.onopen    = () => { document.getElementById('status').textContent = 'streaming'; };
ws.onclose   = () => { document.getElementById('status').textContent = 'disconnected'; };
ws.onerror   = ()  => { document.getElementById('status').textContent = 'error'; };
ws.onmessage = (ev) => {
  const m = MessagePack.decode(new Uint8Array(ev.data));
  applyUpdate(m);
};
</script>
</body>
</html>
"""


async def _index(_request: web.Request) -> web.Response:
    return web.Response(text=_DASHBOARD_HTML, content_type="text/html")


async def _healthcheck(_request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


# ----------------------------------------------------------------------------
# Application factory
# ----------------------------------------------------------------------------


def build_app(*, rate_hz: float, n_layers: int) -> web.Application:
    """Construct the aiohttp application.

    Parameters
    ----------
    rate_hz
        Producer update frequency.
    n_layers
        Layers reported in the per-layer heatmap.

    Returns
    -------
    aiohttp.web.Application
        Configured web application.
    """
    queue = MetricQueue(max_per_subscriber=512)
    trainer = SimulatedTrainer(n_layers=n_layers, rate_hz=rate_hz)
    app = web.Application()
    app["queue"] = queue
    app["trainer"] = trainer
    app.router.add_get("/", _index)
    app.router.add_get("/health", _healthcheck)
    app.router.add_get("/ws", _ws_handler)

    async def _start_producer(app_: web.Application) -> None:
        app_["producer_task"] = asyncio.create_task(trainer.run(queue))

    async def _stop_producer(app_: web.Application) -> None:
        task = app_.get("producer_task")
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app.on_startup.append(_start_producer)
    app.on_cleanup.append(_stop_producer)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--rate-hz", default=5.0, type=float, help="metric update rate")
    parser.add_argument("--n-layers", default=12, type=int)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s | %(levelname)s | %(message)s")
    app = build_app(rate_hz=args.rate_hz, n_layers=args.n_layers)
    _log.info("Serving TensorLens dashboard on http://%s:%d", args.host, args.port)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    # Suppress noisy aiohttp access logs during demo runs.
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)
    # Avoid blocking if the user has stdin redirected.
    try:
        main()
    except KeyboardInterrupt:
        _log.info("Shut down by user")
