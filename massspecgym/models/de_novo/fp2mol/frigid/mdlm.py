"""
Self-contained Masked Diffusion Language Model (MDLM) implementation.

Replaces the bionemo.moco dependency with a standalone ~200 line implementation
matching the MDLM formulation from Sahoo et al. 2024 and FRIGID's usage.

Key operations:
- Log-linear exponential noise schedule: alpha(t) = exp((1-t)*log(alpha_max) + t*log(alpha_min))
- Forward process: independently mask each token with probability 1 - alpha(t)
- Loss: masked-token NLL weighted by -alpha'(t)/alpha(t)
- Confidence-based sampling: unmask most confident positions iteratively
"""

import math

import torch
import torch.nn.functional as F


class LogLinearExpNoiseSchedule:
    """Log-linear exponential noise schedule for MDLM.

    alpha(t) = exp((1 - t) * log(alpha_max) + t * log(alpha_min))
    sigma(t) = -log(alpha(t))

    At t=0: alpha = alpha_max (no masking), at t=1: alpha = alpha_min (nearly all masked).
    """

    def __init__(self, alpha_max: float = 1.0, alpha_min: float = 1e-3):
        self.log_alpha_max = math.log(alpha_max)
        self.log_alpha_min = math.log(alpha_min)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Compute alpha(t) for given time values."""
        log_alpha = (1 - t) * self.log_alpha_max + t * self.log_alpha_min
        return torch.exp(log_alpha)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Compute sigma(t) = -log(alpha(t))."""
        return -((1 - t) * self.log_alpha_max + t * self.log_alpha_min)

    def d_sigma_dt(self, t: torch.Tensor) -> torch.Tensor:
        """Compute d/dt sigma(t) = log(alpha_max) - log(alpha_min).

        For default alpha_max=1: d_sigma/dt = -log(alpha_min) > 0.
        """
        return torch.full_like(t, self.log_alpha_max - self.log_alpha_min)

    def loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the continuous-time NELBO weight: d_sigma/dt / (exp(sigma) - 1).

        This is the correct importance weight from Sahoo et al. 2024 (Eq. 11)
        for the MDLM continuous-time NELBO, matching bionemo's implementation.
        """
        sig = self.sigma(t)
        dsig = self.d_sigma_dt(t)
        return dsig / torch.expm1(sig).clamp(min=1e-8)


class MDLM:
    """Masked Diffusion Language Model.

    Implements the forward masking process, ELBO loss computation, and
    confidence-based iterative unmasking for generation.

    Args:
        mask_token_id: Token ID used for [MASK].
        vocab_size: Size of the token vocabulary.
        noise_schedule: Noise schedule instance.
        sampling_eps: Minimum time value to avoid numerical issues.
    """

    def __init__(
        self,
        mask_token_id: int,
        vocab_size: int,
        noise_schedule: LogLinearExpNoiseSchedule = None,
        sampling_eps: float = 1e-3,
    ):
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.noise_schedule = noise_schedule or LogLinearExpNoiseSchedule()
        self.sampling_eps = sampling_eps
        self._device = torch.device("cpu")

    def to_device(self, device):
        self._device = device

    def sample_time(self, batch_size: int, antithetic: bool = True) -> torch.Tensor:
        """Sample diffusion time t ~ U(eps, 1).

        With antithetic sampling, pairs (t, 1-t+eps) are used for variance reduction.
        """
        if antithetic and batch_size % 2 == 0:
            half = batch_size // 2
            t = torch.rand(half, device=self._device) * (1 - self.sampling_eps) + self.sampling_eps
            t = torch.cat([t, 1 - t + self.sampling_eps], dim=0)
        else:
            t = torch.rand(batch_size, device=self._device) * (1 - self.sampling_eps) + self.sampling_eps
        return t

    def forward_process(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Apply forward masking process.

        Each token is independently replaced with [MASK] with probability 1 - alpha(t).

        Args:
            x0: Clean token IDs [batch, seq_len].
            t: Time values [batch].

        Returns:
            Noisy token IDs with some tokens replaced by mask_token_id.
        """
        alpha_t = self.noise_schedule.alpha(t)  # [batch]
        keep_prob = alpha_t[:, None]  # [batch, 1]
        mask = torch.rand_like(x0.float()) > keep_prob  # True = mask this token
        xt = x0.clone()
        xt[mask] = self.mask_token_id
        return xt

    def loss(
        self,
        logits: torch.Tensor,
        x0: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor = None,
        global_mean: bool = True,
    ) -> torch.Tensor:
        """Compute MDLM training loss (continuous-time NELBO).

        Matches bionemo.moco MDLM.loss(use_weight=True) exactly:
        - Uses substitution parameterization implicitly: loss only at masked positions.
        - Weight: d_sigma/dt / (exp(sigma) - 1), the correct continuous-time NELBO weight
          from Sahoo et al. 2024 (Eq. 11) / https://arxiv.org/pdf/2406.07524.

        Args:
            logits: Model output [batch, seq_len, vocab_size].
            x0: Clean token IDs [batch, seq_len].
            xt: Noisy token IDs [batch, seq_len].
            t: Time values [batch].
            mask: Attention mask [batch, seq_len] (1=valid, 0=padding).
            global_mean: If True, sum all token losses and divide by total token count
                (matching bionemo's global_mean=True behavior).

        Returns:
            Loss tensor (scalar if global_mean, else per-sample [batch]).
        """
        log_probs = F.log_softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]
        nll = F.nll_loss(
            log_probs.view(-1, self.vocab_size),
            x0.view(-1),
            reduction="none",
        ).view_as(x0)  # [batch, seq_len]

        # Subs parameterization: only count loss at masked positions.
        # Non-masked positions would have log_p=0 under subs param, so zeroing
        # them out here is equivalent to bionemo's _subs_parameterization.
        is_masked = xt == self.mask_token_id
        nll = nll * is_masked.float()

        # Apply mask for padding
        if mask is not None:
            nll = nll * mask.float()

        # Continuous-time NELBO weight: d_sigma/dt / (exp(sigma) - 1)
        weight = self.noise_schedule.loss_weight(t)  # [batch]

        # Weight each sample's token NLL sum
        weighted_nll = nll.sum(dim=1) * weight  # [batch]

        if global_mean:
            if mask is not None:
                total_tokens = mask.float().sum()
            else:
                total_tokens = float(x0.numel())
            total_tokens = max(total_tokens, 1.0)
            return weighted_nll.sum() / total_tokens
        else:
            if mask is not None:
                num_tokens = mask.float().sum(dim=1).clamp(min=1)
            else:
                num_tokens = float(x0.size(1))
            return weighted_nll / num_tokens

    def get_num_steps_confidence(self, x: torch.Tensor) -> int:
        """Get number of unmasking steps = number of [MASK] tokens in x."""
        return (x == self.mask_token_id).sum(dim=-1).max().item()

    @torch.no_grad()
    def step_confidence(
        self,
        logits: torch.Tensor,
        x: torch.Tensor,
        step_idx: int,
        num_steps: int,
        temperature: float = 1.0,
        randomness: float = 1.0,
    ) -> torch.Tensor:
        """One confidence-based unmasking step, matching bionemo MDLM exactly.

        1. Apply subs parameterization (copy non-masked tokens).
        2. Sample provisional tokens from softmax(logits / temperature).
        3. Gather per-token confidence = p(sampled_token).
        4. Perturb confidence with annealed Gumbel noise: (log(conf) + noise) / 1.0.
        5. Unmask the most confident position; keep others masked.

        Args:
            logits: Model logits [batch, seq_len, vocab_size].
            x: Current token IDs with masks [batch, seq_len].
            step_idx: Current step index (0-based).
            num_steps: Total number of steps.
            temperature: Softmax temperature for logits.
            randomness: Gumbel noise scale (annealed linearly over steps).

        Returns:
            Updated token IDs with one fewer mask per batch element.
        """
        x_new = x.clone()

        probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)
        preds = torch.distributions.Categorical(probs=probs).sample()  # [batch, seq_len]

        # Confidence = probability of the sampled token
        confidence = probs.gather(-1, preds.unsqueeze(-1)).squeeze(-1)  # [batch, seq_len]

        # Annealed Gumbel noise (decreases linearly from randomness to 0)
        ratio = step_idx / max(num_steps - 1, 1)
        gumbel_noise = -torch.log(-torch.log(
            torch.rand_like(confidence, device=logits.device)
        ))
        gumbel_noise = gumbel_noise * randomness * (1 - ratio)

        # Log-confidence + Gumbel (matching bionemo: log(conf) + noise)
        confidence = torch.log(confidence.clamp(min=1e-8)) + gumbel_noise

        # Only consider masked positions
        is_masked = x == self.mask_token_id
        confidence[~is_masked] = -float("inf")

        # Unmask the single most confident position per batch element
        best_pos = confidence.argmax(dim=-1)  # [batch]
        batch_idx = torch.arange(x.size(0), device=x.device)
        x_new[batch_idx, best_pos] = preds[batch_idx, best_pos]

        return x_new
