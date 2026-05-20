r"""Sparse Autoencoder (SAE) for monosemantic feature recovery.

Implements the Anthropic-style overcomplete SAE (Bricken et al., 2023):

.. math::

    z \;=\; \mathrm{ReLU}(W_{\mathrm{enc}}\,(x - b_{\mathrm{dec}}) + b_{\mathrm{enc}}),
    \qquad \hat{x} \;=\; W_{\mathrm{dec}}\,z + b_{\mathrm{dec}}

with the dictionary expansion :math:`F = m \cdot d`, :math:`m \in [4, 32]`.
The loss combines L2 reconstruction and L1 sparsity:

.. math::

    \mathcal{L}_{\mathrm{SAE}}(x) \;=\; \|x - \hat{x}\|_2^2 \;+\;
    \lambda\,\|z\|_1

Decoder columns are constrained to unit norm via projection after every
optimizer step; this prevents the trivial scale-shifting minimum where
:math:`W_{\mathrm{dec}}` shrinks and :math:`z` grows by the inverse factor.

Dead-feature resampling (re-initializing decoder columns whose features
never activate over the last window of training) is included as an optional
hygiene step.

References
----------
* Bricken, T. et al. (2023). "Towards Monosemanticity: Decomposing Language
  Models with Dictionary Learning." Anthropic.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import nn

from ..utils import MechanisticError, get_logger

__all__ = ["SparseAutoencoder", "SAETrainingConfig", "train_sae"]

_log = get_logger(__name__)


class SparseAutoencoder(nn.Module):
    """Overcomplete sparse autoencoder.

    Parameters
    ----------
    d_model
        Activation dimensionality.
    n_features
        Number of dictionary atoms; should exceed ``d_model``.
    tied_init
        If True, initialize :math:`W_{\\mathrm{enc}}` as the transpose of
        :math:`W_{\\mathrm{dec}}`. Decoupling them during training is now
        standard (Anthropic).
    """

    d_model: int
    n_features: int

    def __init__(self, d_model: int, n_features: int, *, tied_init: bool = True) -> None:
        super().__init__()
        if d_model <= 0 or n_features <= 0:
            raise MechanisticError("d_model and n_features must be positive")
        if n_features <= d_model:
            _log.warning(
                "n_features (%d) <= d_model (%d): SAE will not be overcomplete",
                n_features,
                d_model,
            )
        self.d_model = d_model
        self.n_features = n_features
        # W_enc : (F, d), W_dec : (d, F)
        scale = (1.0 / d_model) ** 0.5
        W_dec_init = torch.randn(d_model, n_features) * scale
        W_dec_init = W_dec_init / W_dec_init.norm(dim=0, keepdim=True).clamp_min(1e-12)
        self.W_dec = nn.Parameter(W_dec_init)
        if tied_init:
            W_enc_init = W_dec_init.T.clone()
        else:
            W_enc_init = torch.randn(n_features, d_model) * scale
        self.W_enc = nn.Parameter(W_enc_init)
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations into the sparse code.

        Parameters
        ----------
        x
            Tensor of shape ``(..., d_model)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(..., n_features)``.
        """
        return torch.relu((x - self.b_dec) @ self.W_enc.T + self.b_enc)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a sparse code back to activation space."""
        return z @ self.W_dec.T + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the autoencoder.

        Returns
        -------
        tuple
            ``(reconstruction, code)``.
        """
        z = self.encode(x)
        return self.decode(z), z

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Project decoder columns onto the unit sphere in :math:`\\mathbb{R}^d`."""
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-12)
        self.W_dec.mul_(1.0 / norms)


@dataclass(frozen=True)
class SAETrainingConfig:
    """Hyperparameters for SAE training.

    Attributes
    ----------
    l1_coefficient
        Coefficient :math:`\\lambda` on the L1 sparsity penalty.
    learning_rate
        Adam learning rate.
    batch_size
        Mini-batch size used to iterate over the activation tensor.
    n_epochs
        Number of passes over the activation set.
    dead_feature_window
        Number of training steps over which to monitor dead features.
    dead_feature_threshold
        Feature activations below this rate are considered dead.
    resample_dead
        If True, periodically reinitialize dead features.
    seed
        RNG seed for batch sampling.
    """

    l1_coefficient: float = 1.0e-3
    learning_rate: float = 1.0e-3
    batch_size: int = 256
    n_epochs: int = 20
    dead_feature_window: int = 500
    dead_feature_threshold: float = 1.0e-5
    resample_dead: bool = True
    seed: int = 0


@dataclass
class SAETrainingResult:
    """Outputs of :func:`train_sae`.

    Attributes
    ----------
    losses
        List of total loss values per logging step.
    reconstruction_losses, sparsity_losses
        Decomposed loss components.
    feature_activation_density
        Final per-feature activation density (length ``n_features``).
    n_dead_features
        Number of features whose density is below the configured threshold.
    """

    losses: list[float]
    reconstruction_losses: list[float]
    sparsity_losses: list[float]
    feature_activation_density: torch.Tensor
    n_dead_features: int


def _iterate_batches(
    activations: torch.Tensor,
    batch_size: int,
    seed: int,
) -> Iterator[torch.Tensor]:
    """Yield shuffled batches over an activation tensor.

    Parameters
    ----------
    activations
        Tensor of shape ``(N, d_model)``.
    batch_size
        Mini-batch size.
    seed
        RNG seed.

    Yields
    ------
    torch.Tensor
        Shuffled batches of shape ``(batch_size, d_model)`` (last batch may
        be smaller).
    """
    n = activations.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    for start in range(0, n, batch_size):
        idx = perm[start : start + batch_size]
        yield activations[idx]


def train_sae(
    activations: torch.Tensor,
    sae: SparseAutoencoder,
    config: SAETrainingConfig,
    *,
    device: torch.device | str = "cpu",
) -> SAETrainingResult:
    """Train a sparse autoencoder on a tensor of activations.

    Parameters
    ----------
    activations
        Tensor of shape ``(N, d_model)``.
    sae
        SAE instance.
    config
        Training hyperparameters.
    device
        Compute device.

    Returns
    -------
    SAETrainingResult
        Training trace and final diagnostics.

    Raises
    ------
    MechanisticError
        On shape mismatch with the SAE.
    """
    if activations.dim() != 2 or activations.shape[1] != sae.d_model:
        raise MechanisticError(
            f"activations must be (N, {sae.d_model}); got {tuple(activations.shape)}"
        )
    sae = sae.to(device)
    activations = activations.to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=config.learning_rate)
    losses: list[float] = []
    recon_losses: list[float] = []
    spars_losses: list[float] = []

    feature_fires = torch.zeros(sae.n_features, device=device)
    window_counter = 0
    step = 0

    for epoch in range(config.n_epochs):
        for batch in _iterate_batches(activations, config.batch_size, config.seed + epoch):
            opt.zero_grad(set_to_none=True)
            recon, code = sae(batch)
            recon_loss = ((batch - recon) ** 2).sum(dim=-1).mean()
            sparsity_loss = code.abs().sum(dim=-1).mean()
            loss = recon_loss + config.l1_coefficient * sparsity_loss
            loss.backward()
            # Project gradient component that would change ||W_dec[:,k]||.
            # Standard hygiene: remove gradient parallel to decoder column.
            with torch.no_grad():
                if sae.W_dec.grad is not None:
                    proj = (sae.W_dec.grad * sae.W_dec).sum(dim=0, keepdim=True)
                    sae.W_dec.grad.sub_(proj * sae.W_dec)
            opt.step()
            sae.normalize_decoder()
            # Bookkeeping
            with torch.no_grad():
                feature_fires += (code > 0).float().sum(dim=0)
                window_counter += batch.shape[0]
            losses.append(float(loss.detach()))
            recon_losses.append(float(recon_loss.detach()))
            spars_losses.append(float(sparsity_loss.detach()))
            step += 1
            if (
                config.resample_dead
                and step % config.dead_feature_window == 0
                and window_counter > 0
            ):
                density = feature_fires / max(window_counter, 1)
                dead = density < config.dead_feature_threshold
                n_dead = int(dead.sum().item())
                if n_dead > 0:
                    _resample_dead_features(sae, dead, activations)
                    _log.info("Resampled %d dead features at step %d", n_dead, step)
                feature_fires.zero_()
                window_counter = 0
        _log.info("Epoch %d/%d  loss=%.5f", epoch + 1, config.n_epochs, losses[-1])

    # Final density on the full set
    with torch.no_grad():
        all_codes = sae.encode(activations)
        density_final = (all_codes > 0).float().mean(dim=0).cpu()
        n_dead_final = int((density_final < config.dead_feature_threshold).sum().item())

    return SAETrainingResult(
        losses=losses,
        reconstruction_losses=recon_losses,
        sparsity_losses=spars_losses,
        feature_activation_density=density_final,
        n_dead_features=n_dead_final,
    )


@torch.no_grad()
def _resample_dead_features(
    sae: SparseAutoencoder,
    dead_mask: torch.Tensor,
    activations: torch.Tensor,
) -> None:
    """Reinitialize decoder columns / encoder rows for dead features.

    Strategy (simplified Anthropic recipe): replace dead decoder columns with
    randomly sampled, normalized activation vectors from the training set,
    and zero the corresponding encoder rows + biases.

    Parameters
    ----------
    sae
        Sparse autoencoder.
    dead_mask
        Boolean tensor of shape ``(n_features,)``.
    activations
        Activation tensor sampled from for replacement vectors.
    """
    n_dead = int(dead_mask.sum().item())
    if n_dead == 0:
        return
    idx = torch.randint(0, activations.shape[0], (n_dead,), device=activations.device)
    new_atoms = activations[idx]
    new_atoms = new_atoms - new_atoms.mean(dim=0, keepdim=True)
    new_atoms = new_atoms / new_atoms.norm(dim=1, keepdim=True).clamp_min(1e-12)
    sae.W_dec[:, dead_mask] = new_atoms.T
    sae.W_enc[dead_mask, :] = new_atoms * 0.2  # smaller encoder magnitude
    sae.b_enc[dead_mask] = 0.0
