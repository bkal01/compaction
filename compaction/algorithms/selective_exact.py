"""
Selective Exact KV cache compaction algorithm.

Key selection is a two-pass procedure that achieves near-exact attention score
ranking at a fraction of the cost of running all queries against all keys:

  1. Rough pass: sample n_rough random queries, compute softmax scores for all T keys.
  2. Restrict to the top k_factor*t candidates by rough score.
  3. Exact pass: compute softmax scores with ALL queries, over candidates only.
  4. Return top-t from exact scores.

Score aggregation across queries is controlled by score_method: 'rms' (default),
'mean', or 'max'.

Beta fitting (redistribute_uniform) reuses the exact-pass weights over all queries —
Z_cand (sum over k_factor*t candidates) serves as a proxy for Z_full, at no extra cost.
C2 is either taken directly from V[selected_indices] or fitted via least squares.
"""

import torch
from typing import Tuple
from .base import CompactionAlgorithm


class SelectiveExactCompaction(CompactionAlgorithm):
    """
    Selective Exact KV cache compaction.

    Recommended configuration: k_factor=3, n_rough=256, score_method='rms',
    beta='redistribute_uniform', c2='direct' — ~0.82 key overlap with full-RMS
    ranking at 3.1× speedup.
    Direct C2 outperforms LSQ fitting when n_rough << t; see C2_FITTING_ANALYSIS.md.
    """

    def __init__(
        self,
        k_factor: int = 3,
        n_rough: int = 256,
        score_method: str = 'rms',
        beta_method: str = 'redistribute_uniform',
        c2_method: str = 'direct',
        nnls_iters: int = 0,
        nnls_lower_bound: float = None,
        nnls_upper_bound: float = None,
        c2_ridge_lambda: float = 0,
        c2_solver: str = 'lstsq',
        c2_ridge_scale: str = 'spectral',
    ):
        """
        Parameters
        ----------
        k_factor : int
            Candidate expansion factor. Rough pass keeps top k_factor*t keys.
            k=3 → ~3.1× speedup, ~0.82 overlap; k=5 → ~1.9× speedup, ~0.90 overlap.
        n_rough : int
            Number of random queries for the rough pass (≤ 512). Doubling from
            256 to 512 adds ~1 pp overlap; bottleneck is k_factor, not n_rough.
        score_method : str
            How to aggregate per-query softmax attention weights into a single
            key score: 'rms' (default), 'mean', or 'max'.
        beta_method : str
            'redistribute_uniform' to uniformly redistribute estimated missing
            attention mass (default), 'nnls' for the clamped-least-squares
            partition mass matching solution, or 'zero' to set all beta values
            to zero.
        c2_method : str
            'direct' to set C2 = V[selected_indices] (default; recommended),
            or 'lsq' to fit C2 via least squares against the rough-pass queries.
        nnls_iters : int
            PGD iterations for beta_method='nnls'. 0 uses clamped least squares.
        nnls_lower_bound : float, optional
            Lower bound for exp(beta) when beta_method='nnls'. Default 1e-12.
        nnls_upper_bound : float, optional
            Upper bound for exp(beta) when beta_method='nnls'. Default None.
        c2_ridge_lambda : float
            Ridge regularization for c2_method='lsq' (default: 0).
        c2_solver : str
            Solver for c2_method='lsq': 'lstsq', 'pinv', or 'cholesky'.
        c2_ridge_scale : str
            Ridge scaling for c2_method='lsq': 'spectral', 'frobenius', or 'fixed'.
        """
        if not isinstance(k_factor, int) or k_factor < 1:
            raise ValueError(f"k_factor must be a positive int, got {k_factor!r}")
        if not isinstance(n_rough, int) or n_rough < 1:
            raise ValueError(f"n_rough must be a positive int, got {n_rough!r}")
        if n_rough > 512:
            raise ValueError(f"n_rough must be <= 512, got {n_rough}")
        if score_method not in ['rms', 'mean', 'max']:
            raise ValueError(f"score_method must be 'rms', 'mean', or 'max', got '{score_method}'")
        if beta_method not in ['redistribute_uniform', 'nnls', 'zero']:
            raise ValueError(
                "beta_method must be 'redistribute_uniform', 'nnls', or "
                f"'zero', got '{beta_method}'"
            )
        if c2_method not in ['direct', 'lsq']:
            raise ValueError(f"c2_method must be 'direct' or 'lsq', got '{c2_method}'")
        if c2_solver not in ['lstsq', 'pinv', 'cholesky']:
            raise ValueError(
                f"c2_solver must be 'lstsq', 'pinv', or 'cholesky', got '{c2_solver}'"
            )
        if c2_ridge_scale not in ['spectral', 'frobenius', 'fixed']:
            raise ValueError(
                f"c2_ridge_scale must be 'spectral', 'frobenius', or 'fixed', got '{c2_ridge_scale}'"
            )
        self.k_factor = k_factor
        self.n_rough = n_rough
        self.score_method = score_method
        self.beta_method = beta_method
        self.c2_method = c2_method
        self.nnls_iters = nnls_iters
        self.nnls_lower_bound = nnls_lower_bound
        self.nnls_upper_bound = nnls_upper_bound
        self.c2_ridge_lambda = c2_ridge_lambda
        self.c2_solver = c2_solver
        self.c2_ridge_scale = c2_ridge_scale

    def name(self) -> str:
        return "SelectiveExact"

    def compute_compacted_cache(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int,
        attention_bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Compute compacted cache via Selective Exact RMS (SER) key selection.

        Parameters
        ----------
        K : Tensor, shape (T, d)
        V : Tensor, shape (T, d)
        queries : Tensor, shape (n, d)
            All available query vectors (context-prefill queries).
        t : int
            Number of keys to retain in the compacted cache.
        attention_bias : Tensor, optional
            Additive per-key bias for the original cache (broadcastable to (T,)).

        Returns
        -------
        C1 : Tensor, shape (t, d)
        beta : Tensor, shape (t,)
        C2 : Tensor, shape (t, d)
        indices : list of int
        """
        T, d = K.size()
        n, _ = queries.size()
        device = K.device
        dtype_param = K.dtype
        inv_sqrt_d = (1.0 / d) ** 0.5

        bias32 = None if attention_bias is None else attention_bias.to(torch.float32)

        # --- Rough pass: n_rough random queries score all T keys ---
        n_rough = min(self.n_rough, n)
        rough_idx = torch.randperm(n, device=device)[:n_rough]
        rough_queries = queries[rough_idx].to(dtype_param)

        rough_scores32 = (rough_queries @ K.T).to(torch.float32) * inv_sqrt_d
        if bias32 is not None:
            rough_scores32 = rough_scores32 + bias32
        rough_row_max = rough_scores32.max(dim=1, keepdim=True).values
        rough_w = torch.exp(rough_scores32 - rough_row_max)
        rough_w = rough_w / rough_w.sum(dim=1, keepdim=True)
        rough_key_scores = self._score_weights(rough_w)  # (T,)

        # --- Exact pass: all queries over top k_factor*t candidates ---
        n_cand = min(self.k_factor * t, T)
        cand_idx = torch.topk(rough_key_scores, n_cand, largest=True).indices
        K_cand = K[cand_idx]

        exact_scores32 = (queries.to(dtype_param) @ K_cand.T).to(torch.float32) * inv_sqrt_d
        if bias32 is not None:
            exact_scores32 = exact_scores32 + bias32[cand_idx]
        exact_row_max = exact_scores32.max(dim=1, keepdim=True).values
        exact_w = torch.exp(exact_scores32 - exact_row_max)
        exact_Z_cand = exact_w.sum(dim=1, keepdim=True)  # (n, 1) — reused for beta
        exact_w = exact_w / exact_Z_cand
        exact_key_scores = self._score_weights(exact_w)  # (n_cand,)

        # sel_in_cand: positions within candidates; reused below for beta.
        _, sel_in_cand = torch.topk(exact_key_scores, t, largest=True)
        sel_idx = cand_idx[sel_in_cand]
        C1 = K[sel_idx]
        indices = sel_idx.cpu().tolist()

        # --- Beta ---
        if self.beta_method == 'zero':
            beta32 = torch.zeros(t, dtype=torch.float32, device=device)
        elif self.beta_method == 'redistribute_uniform':
            # All n queries, Z_cand as proxy for Z_full — no extra matmul.
            sel_unnorm = exact_w[:, sel_in_cand] * exact_Z_cand  # (n, t)
            Z_sel = sel_unnorm.sum(dim=1)  # (n,)
            missing_mass = (exact_Z_cand.squeeze(1) - Z_sel).clamp_min(0.0).mean()
            selected_base_mass = sel_unnorm.mean(dim=0)  # (t,)
            beta32 = torch.log(
                (selected_base_mass + missing_mass / t).clamp_min(1e-12)
                / selected_base_mass.clamp_min(1e-12)
            )
        else:  # 'nnls'
            beta32 = self._beta_nnls(K, C1, rough_queries, bias32)

        beta = beta32.to(dtype_param)

        # --- C2 ---
        if self.c2_method == 'direct':
            C2 = self._direct_C2(C1, K, V, indices)
        else:  # 'lsq'
            C2 = self._compute_C2(
                C1, beta, K, V, rough_queries,
                attention_bias=bias32,
                ridge_lambda=self.c2_ridge_lambda,
                solver=self.c2_solver,
                ridge_scale=self.c2_ridge_scale,
            )

        return C1, beta, C2, indices

    def _score_weights(self, w: torch.Tensor) -> torch.Tensor:
        """Aggregate softmax attention weights (n, T) → per-key score (T,)."""
        if self.score_method == 'rms':
            return (w ** 2).mean(dim=0).sqrt()
        elif self.score_method == 'mean':
            return w.mean(dim=0)
        else:  # 'max'
            return w.max(dim=0).values

    def _beta_nnls(
        self,
        K: torch.Tensor,
        C1: torch.Tensor,
        rough_queries: torch.Tensor,
        bias32: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fit beta by matching full-attention partition mass over rough-pass queries
        using non-negative least squares / clamped least squares.
        """
        _, d = K.size()
        inv_sqrt_d = (1.0 / d) ** 0.5
        dtype_param = K.dtype
        fq = rough_queries.to(dtype_param)

        full_scores = (fq @ K.T).to(torch.float32) * inv_sqrt_d
        if bias32 is not None:
            full_scores = full_scores + bias32
        compacted_scores = (fq @ C1.T).to(torch.float32) * inv_sqrt_d

        ref = torch.maximum(
            full_scores.max(dim=1, keepdim=True).values,
            compacted_scores.max(dim=1, keepdim=True).values,
        )
        target = torch.exp((full_scores - ref).clamp(min=-60.0, max=60.0)).sum(dim=1)
        M = torch.exp((compacted_scores - ref).clamp(min=-60.0, max=60.0))

        weights = self._nnls_pg(
            M, target, self.nnls_iters, self.nnls_lower_bound, self.nnls_upper_bound
        )
        return torch.log(weights)
