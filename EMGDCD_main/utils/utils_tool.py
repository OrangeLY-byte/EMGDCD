import torch
from torch import Tensor, nn
import warnings
from torch.distributions import Bernoulli, kl_divergence
import torch.nn.functional as F

def gumbel_softmax(logits: Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> Tensor:
    r"""
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class spars_loss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-8, prior_prob=1e-6, divergence='jsd'):
        """
        KL or Jensen-Shannon divergence between Bernoulli(p) and the prior.

        Args:
            reduction: 'mean', 'sum', or 'none'.
            eps: Probability clamp threshold for numerical stability.
            prior_prob: Prior edge probability, strictly between 0 and 1.
            divergence: 'kl' or 'jsd' (case-insensitive); defaults to 'kl'.
        """
        super().__init__()

        if not isinstance(divergence, str) or divergence.lower() not in ('kl', 'jsd'):
            raise ValueError("divergence must be 'kl' or 'jsd'")
        self.divergence = divergence.lower()

        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        if not 0 < eps < 0.5:
            raise ValueError("eps must be between 0 and 0.5")
        if not 0 < prior_prob < 1:
            raise ValueError("prior_prob must be between 0 and 1")

        self.reduction = reduction
        self.eps = eps
        self.prior_prob = prior_prob

    def forward(self, p):
        """
        Args:
            p: Edge probabilities with shape (B, S, N, N).

        Returns:
            Scalar for 'mean' or 'sum';
            tensor of shape (N, N) for 'none'.
        """
        # Ensure the clamp is representable in the input dtype.
        eps = max(self.eps, torch.finfo(p.dtype).eps)

        p_safe = p.clamp(eps, 1 - eps)
        q_safe = torch.full_like(p, self.prior_prob).clamp(eps, 1 - eps)

        P_dist = Bernoulli(probs=p_safe)
        Q_dist = Bernoulli(probs=q_safe)

        if self.divergence == 'kl':
            loss = kl_divergence(P_dist, Q_dist)
        else:
            M_dist = Bernoulli(probs=0.5 * (p_safe + q_safe))
            loss = 0.5 * (
                kl_divergence(P_dist, M_dist)
                + kl_divergence(Q_dist, M_dist)
            )
        loss_reduced = loss.mean(dim=(0, 1))   # (N, N)

        if self.reduction == 'mean':
            return loss_reduced.mean()
        elif self.reduction == 'sum':
            return loss_reduced.sum()
        else:
            return loss_reduced