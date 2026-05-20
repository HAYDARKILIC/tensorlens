# Mathematical Appendix

This document collects the closed-form derivations behind each TensorLens
diagnostic. Knowing *why* a formula has the shape it does is the difference
between using a library and being able to debug it.

---

## 1. Anisotropy and effective rank

### 1.1 Empirical anisotropy

For a set of vectors $\mathcal{H} = \{h_1, \ldots, h_N\} \subset \mathbb{R}^d$,
the **anisotropy index** (Ethayarajh, 2019) is the expected pairwise cosine
similarity:

$$\mathcal{A}(\mathcal{H}) \;=\; \frac{1}{N(N-1)} \sum_{i \neq j}
\frac{\langle h_i, h_j\rangle}{\|h_i\|_2 \|h_j\|_2}$$

If $h_i$ are i.i.d. samples from a spherically symmetric distribution then
$\mathbb{E}\,\mathcal{A} = 0$. Empirically, transformer hidden states yield
$\mathcal{A} \in [0.4, 0.9]$ — a clear violation of isotropy.

### 1.2 Closed-form anisotropy for a planted cone

Consider the generative model

$$h_i \;=\; \alpha u + \xi_i, \qquad \xi_i \sim \mathcal{N}(0, \sigma^2 I_d),
\quad \|u\| = 1$$

Then $\|h_i\|^2 = \alpha^2 + 2\alpha u^\top \xi_i + \|\xi_i\|^2$ and
$\langle h_i, h_j\rangle = \alpha^2 + \alpha u^\top(\xi_i + \xi_j) +
\xi_i^\top \xi_j$. Taking expectations and assuming $\sigma^2 d \ll \alpha^2$:

$$\mathbb{E}\,\frac{\langle h_i, h_j\rangle}{\|h_i\|\|h_j\|} \;\approx\;
\frac{\alpha^2}{\alpha^2 + \sigma^2 d \cdot \text{(small)}}$$

so to hit a target anisotropy $\mathcal{A}^*$ we set $\alpha = \sigma
\sqrt{\mathcal{A}^* / (1 - \mathcal{A}^*)}$. This is the closed form used by
`synth_anisotropic_hidden_states`.

### 1.3 Effective rank

For a matrix $H \in \mathbb{R}^{N \times d}$ with singular values $\sigma_1
\geq \ldots \geq \sigma_r > 0$, define $p_k = \sigma_k^2 / \sum_j \sigma_j^2$
— the energy distribution. The **entropy-based effective rank** (Roy &
Vetterli, 2007) is

$$r_\mathrm{eff}(H) \;=\; \exp\!\Bigl(-\sum_{k=1}^{r} p_k \log p_k\Bigr)$$

For a uniform spectrum $r_\mathrm{eff} = r$; for a rank-1 spectrum
$r_\mathrm{eff} = 1$. The exponentiated Shannon entropy makes the value
interpretable as a "number of effectively used dimensions".

### 1.4 Gini concentration

The Gini coefficient applied to the sorted energy vector $(p_1, \ldots, p_d)$
with $p_1 \geq \cdots \geq p_d$:

$$G \;=\; \frac{2 \sum_{k=1}^{d} k\,p_k}{d \sum_k p_k} - \frac{d+1}{d}$$

Bounded $G \in [0, 1)$. $G = 0$ for a uniform spectrum; $G \to (d-1)/d$ for
rank-1.

---

## 2. t-SNE

### 2.1 High-dim affinities

Conditional probability of $j$ given $i$:

$$p_{j|i} \;=\; \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}
                       {\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$$

Symmetrized joint:

$$p_{ij} \;=\; \frac{p_{j|i} + p_{i|j}}{2N}$$

### 2.2 Bandwidth calibration

$\sigma_i$ is set so the conditional distribution $P_i$ has a target
**perplexity** $u$:

$$2^{H(P_i)} \;=\; u, \qquad H(P_i) = -\sum_j p_{j|i} \log_2 p_{j|i}$$

Solved by bisection on $\beta_i = 1/(2\sigma_i^2)$.

### 2.3 Low-dim affinities

Student-t kernel with 1 degree of freedom (heavy-tailed → resists the
crowding problem):

$$q_{ij} \;=\; \frac{(1 + \|y_i - y_j\|^2)^{-1}}
                    {\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}$$

### 2.4 Gradient

$$\frac{\partial \mathcal{L}}{\partial y_i} \;=\; 4 \sum_{j} (p_{ij} - q_{ij})\,
(y_i - y_j) \,(1 + \|y_i - y_j\|^2)^{-1}$$

---

## 3. UMAP — the simplified cross-entropy

### 3.1 Per-point local connectivity

For each point $i$:

$$\rho_i \;=\; \min_{j \in \mathcal{N}_k(i)} d(x_i, x_j)$$

The kernel scale $\sigma_i$ is calibrated so

$$\sum_{j \in \mathcal{N}_k(i)} \exp\!\Bigl(-\frac{\max(0, d_{ij} - \rho_i)}
                                        {\sigma_i}\Bigr) \;=\; \log_2 k$$

### 3.2 Symmetrization

$$v_{ij}^{(h)} = \exp\!\Bigl(-\frac{\max(0, d_{ij} - \rho_i)}{\sigma_i}\Bigr),
\qquad w_{ij} = v_{ij}^{(h)} + v_{ji}^{(h)} - v_{ij}^{(h)} v_{ji}^{(h)}$$

(probabilistic OR — a fuzzy union).

### 3.3 Low-dim weights

$$v_{ij} = \bigl(1 + a\,\|y_i - y_j\|^{2b}\bigr)^{-1}$$

with $(a, b)$ calibrated by least-squares fit to the target piecewise
exponential. Cross-entropy loss:

$$\mathcal{L} \;=\; \sum_{i \neq j} w_{ij} \log\frac{w_{ij}}{v_{ij}}
                                  + (1 - w_{ij}) \log\frac{1 - w_{ij}}{1 - v_{ij}}$$

---

## 4. Intrinsic dimension

### 4.1 TwoNN

Under a locally uniform density assumption, the ratio $\mu_i = r_{i,2} /
r_{i,1}$ satisfies

$$F(\mu) \;=\; 1 - \mu^{-d}$$

so

$$-\log(1 - F(\mu)) \;=\; d \log \mu$$

A linear fit gives $\hat d$.

### 4.2 MLE

Given the $k$ nearest neighbors with distances $r_{i,1} \leq \ldots \leq
r_{i,k}$:

$$\hat d_i \;=\; \biggl[\frac{1}{k-1} \sum_{j=1}^{k-1}
        \log\frac{r_{i,k}}{r_{i,j}}\biggr]^{-1}$$

The global estimate is the mean over all $i$.

---

## 5. Hessian curvature

### 5.1 Hessian-vector product

For loss $\mathcal{L}(\theta)$ and a direction $v$:

$$H v \;=\; \nabla_\theta\bigl(\nabla_\theta \mathcal{L}(\theta)^\top v\bigr)$$

— a double backward, $O(\text{one forward + one backward})$ in time.

### 5.2 Power iteration

$$v_{t+1} = \frac{H v_t}{\|H v_t\|}, \qquad
\lambda_{\max} \;\approx\; v_t^\top H v_t \;=\; \|H v_t\|$$

Converges geometrically with rate $|\lambda_2 / \lambda_1|^t$.

### 5.3 Lanczos

Iteratively builds the Krylov basis $\mathcal{K}_m = \mathrm{span}(v, Hv,
\ldots, H^{m-1} v)$. The projection $T_m = Q_m^\top H Q_m$ is tridiagonal
with diagonal $\alpha_i$ and off-diagonal $\beta_i$. The extreme eigenvalues
of $T_m$ converge to those of $H$ with full re-orthogonalization at each
step.

### 5.4 Edge of Stability

For gradient descent at step size $\eta$,

$$\theta_{t+1} - \theta_t \;=\; -\eta \nabla \mathcal{L}(\theta_t)$$

is locally stable iff $\eta \lambda_{\max}(H_t) < 2$. Cohen et al. (2021)
showed neural networks operate at the boundary

$$\lambda_{\max}(\theta_t) \;\to\; \frac{2}{\eta}$$

after a progressive sharpening phase.

---

## 6. Roofline model

### 6.1 Definition

$$P_{\mathrm{attainable}}(I) \;=\; \min\!\bigl( P_{\mathrm{peak}}, \beta \cdot I \bigr)$$

with arithmetic intensity $I = \mathrm{FLOPs} / \mathrm{bytes}$. The **ridge
point** is $I^* = P_{\mathrm{peak}} / \beta$.

### 6.2 Multi-tier extension

$$P(I) \;=\; \min\!\Bigl( P_{\mathrm{peak}}, \min_k \beta_k I_k \Bigr)$$

with $I_k = \mathrm{FLOPs} / B_k$ for each memory tier $k$.

### 6.3 Worked examples

| Kernel              | FLOPs               | Bytes (DRAM)        | $I$ (FLOP/B)  | Regime on A100        |
|---------------------|---------------------|---------------------|---------------|------------------------|
| SAXPY               | $2N$                | $12 N$              | $0.17$        | memory-bound           |
| Matmul $M = N = K$  | $2 N^3$             | $4 \cdot 3 N^2$     | $N / 6$       | compute-bound for $N \gtrsim 1000$ |
| Attention (no fuse) | $4 H T^2 d_h$       | $2 H T (T + 2 d_h)$ | $\sim 2 d_h$  | memory-bound           |
| Flash attention     | $4 H T^2 d_h$       | $2 H T \cdot d_h$   | $\sim 2 T$    | compute-bound          |

---

## 7. Sparse Autoencoder

### 7.1 Model

$$z = \mathrm{ReLU}(W_e (x - b_d) + b_e), \qquad \hat x = W_d z + b_d$$

with $\|W_d[:, k]\|_2 = 1$ enforced by projection.

### 7.2 Loss

$$\mathcal{L}_\mathrm{SAE}(x) = \|x - \hat x\|_2^2 + \lambda \|z\|_1$$

### 7.3 Gradient projection

To keep decoder columns on the unit sphere, the gradient $g = \nabla
\mathcal{L}$ is projected:

$$\tilde g_k \;=\; g_k - (g_k^\top W_d[:, k])\, W_d[:, k]$$

then a normal Adam step is taken, followed by exact renormalization.

### 7.4 Dead feature resampling

A feature $k$ is **dead** if $\mathbb{P}[z_k > 0] < \epsilon$ over a sliding
window. Replace its column by a renormalized sample from the activation set:

$$W_d[:, k] \;\leftarrow\; \frac{x_{(i)} - \mathbb{E}\,x}{\|x_{(i)} - \mathbb{E}\,x\|}$$

with $i$ chosen uniformly at random.

---

## References

* Bricken, T. et al. (2023). *Towards Monosemanticity: Decomposing Language
  Models with Dictionary Learning.* Anthropic.
* Cohen, J. et al. (2021). *Gradient Descent on Neural Networks Typically
  Occurs at the Edge of Stability.* ICLR.
* Elhage, N. et al. (2021). *A Mathematical Framework for Transformer Circuits.*
  Anthropic.
* Ethayarajh, K. (2019). *How Contextual are Contextualized Word
  Representations?* EMNLP.
* Facco, E. et al. (2017). *Estimating the intrinsic dimension of datasets.*
  Sci. Reports.
* Levina, E., Bickel, P. J. (2004). *Maximum Likelihood Estimation of
  Intrinsic Dimension.* NeurIPS.
* Li, H. et al. (2018). *Visualizing the Loss Landscape of Neural Nets.*
  NeurIPS.
* McInnes, L., Healy, J., Melville, J. (2018). *UMAP: Uniform Manifold
  Approximation and Projection.* arXiv:1802.03426.
* Olsson, C. et al. (2022). *In-context Learning and Induction Heads.*
  Anthropic.
* Pearlmutter, B. (1994). *Fast Exact Multiplication by the Hessian.*
  Neural Computation.
* Roy, O., Vetterli, M. (2007). *The effective rank: A measure of effective
  dimensionality.* EUSIPCO.
* van der Maaten, L., Hinton, G. (2008). *Visualizing Data using t-SNE.*
  JMLR.
* Williams, S., Waterman, A., Patterson, D. (2009). *Roofline: an insightful
  visual performance model for multicore architectures.* CACM.
