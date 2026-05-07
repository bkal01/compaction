"""
Highest *Expected* Attention Keys KV cache compaction algorithm.

Similar to `highest_attention_keys.py`, but rather than explicitly computing
attention scores for each query-key pair (which is incompatible with FlashAttention
as we would need to materialize the full attention matrix), we compute the expected
attention instead (see https://arxiv.org/pdf/2510.00636).
"""

import torch
import torch.nn.functional as F
from typing import Tuple
from .base import CompactionAlgorithm


class HighestExpectedAttentionKeysCompaction(CompactionAlgorithm):
    """Select keys with highest average expected attention scores."""

    def __init__(self, nnls_iters: int = 0, nnls_lower_bound: float = None, nnls_upper_bound: float = None,
                 c2_method: str = 'taylor_lsq', beta_method: str = 'redistribute_uniform',
                 c2_ridge_lambda: float = 0, c2_solver: str = 'lstsq', c2_ridge_scale: str = 'spectral',
                 pooling: str = None, kernel_size: int = 7):
        """
        Parameters
        ----------
        nnls_iters : int
            Number of projected gradient descent iterations for
            beta_method='nnls' and beta_method='taylor_nnls'. If 0, uses least
            squares with clamping.
        nnls_lower_bound : float, optional
            Lower bound for exp(beta) in beta_method='nnls' and
            beta_method='taylor_nnls'. If None, uses 1e-12.
        nnls_upper_bound : float, optional
            Optional upper bound for exp(beta) in beta_method='nnls' and
            beta_method='taylor_nnls'.
        c2_method : str
            Method to compute C2: 'taylor_lsq' for Taylor least squares around
            the query mean (default), 'lsq' for underdetermined expected-output
            least squares, or 'direct' to reuse values from selected key positions.
        beta_method : str, optional
            Method to compute beta: 'zero' to set all beta=0,
            'redistribute_uniform' to uniformly redistribute estimated missing
            mass (default), 'nnls' for the underdetermined expected-mass
            non-negative least-squares/clamped weight solution, or 'taylor_nnls' to match the
            local partition function and first derivative at the query mean.
        c2_ridge_lambda : float
            Regularization parameter for C2 ridge regression (default: 0).
        c2_solver : str
            Solver to use for C2: 'pinv', 'cholesky', or 'lstsq' (default: 'lstsq').
        c2_ridge_scale : str
            How to scale ridge_lambda: 'spectral', 'frobenius', or 'fixed' (default: 'spectral').
        pooling : str, optional
            Pooling method to apply to attention scores: 'avgpool', 'maxpool', or None (default: None).
        kernel_size : int
            Kernel size for pooling operation (default: 7).
        """
        self.nnls_iters = nnls_iters
        self.nnls_lower_bound = nnls_lower_bound
        self.nnls_upper_bound = nnls_upper_bound
        if c2_method not in ['lsq', 'taylor_lsq', 'direct']:
            raise ValueError(
                "c2_method must be 'lsq', 'taylor_lsq', or 'direct', "
                f"got '{c2_method}'"
            )
        self.c2_method = c2_method
        if c2_solver not in ['lstsq', 'pinv', 'cholesky']:
            raise ValueError(
                "c2_solver must be 'lstsq', 'pinv', or 'cholesky', "
                f"got '{c2_solver}'"
            )
        if c2_ridge_scale not in ['spectral', 'frobenius', 'fixed']:
            raise ValueError(
                "c2_ridge_scale must be 'spectral', 'frobenius', or 'fixed', "
                f"got '{c2_ridge_scale}'"
            )
        if beta_method not in ['zero', 'redistribute_uniform', 'nnls', 'taylor_nnls']:
            raise ValueError(
                "beta_method must be 'zero', 'redistribute_uniform', 'nnls', or "
                "'taylor_nnls', "
                f"got '{beta_method}'"
            )
        self.beta_method = beta_method
        self.c2_ridge_lambda = c2_ridge_lambda
        self.c2_solver = c2_solver
        self.c2_ridge_scale = c2_ridge_scale
        if pooling is not None and pooling not in ['avgpool', 'maxpool']:
            raise ValueError(f"pooling must be 'avgpool', 'maxpool', or None, got '{pooling}'")
        self.pooling = pooling
        self.kernel_size = kernel_size

    def name(self) -> str:
        return "HighestExpectedAttentionKeys"

    def compute_compacted_cache(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int,
        attention_bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Compute compacted cache using highest attention key selection.

        Parameters
        ----------
        K : Tensor, shape (T, d)
            Original key matrix
        V : Tensor, shape (T, d)
            Original value matrix
        queries : Tensor, shape (n, d)
            Query samples for training
        t : int
            Compacted size (number of keys to select)
        attention_bias : Tensor, optional
            Additive per-key attention bias for the original cache (broadcastable to (T,)).
        Returns
        -------
        C1 : Tensor, shape (t, d)
            Compacted keys
        beta : Tensor, shape (t,)
            Bias terms
        C2 : Tensor, shape (t, d)
            Compacted values
        indices : list of int
            Indices of selected keys
        """
        # Select keys based on Gaussian expected exp-scores.
        C1, beta, indices = self._select_keys_highest_expected_attention(K, queries, t, attention_bias)

        if self.c2_method == "direct":
            C2 = self._direct_C2(C1, K, V, indices)
            return C1, beta, C2, indices

        mu = queries.to(torch.float32).mean(dim=0)
        sigma = queries.to(torch.float32).T.cov()

        if self.c2_method == "lsq":
            C2 = self._compute_C2_via_expected_lsq(
                mu=mu,
                sigma=sigma,
                K=K,
                V=V,
                C1=C1,
                beta=beta,
                attention_bias=attention_bias,
                solver=self.c2_solver,
                ridge_lambda=self.c2_ridge_lambda,
                ridge_scale=self.c2_ridge_scale,
            )
        elif self.c2_method == "taylor_lsq":
            # Compute compacted values by matching value + first derivative at q=mu.
            C2 = self._compute_C2_via_expectation_with_method(
                mu=mu,
                K=K,
                V=V,
                C1=C1,
                beta=beta,
                attention_bias=attention_bias,
                solver=self.c2_solver,
                ridge_lambda=self.c2_ridge_lambda,
                ridge_scale=self.c2_ridge_scale,
            )
        else:
            raise ValueError(
                f"Unsupported c2_method '{self.c2_method}' for HighestExpectedAttentionKeysCompaction"
            )

        return C1, beta, C2, indices

    def _select_keys_highest_expected_attention(
        self,
        K: torch.Tensor,
        queries: torch.Tensor,
        t: int,
        attention_bias: torch.Tensor = None,
    ):
        """
        Select t keys from K with highest Gaussian expected exp-scores.

        Parameters
        ----------
        K : Tensor, shape (T, d)
            Original key matrix.
        queries : Tensor, shape (n, d)
            Sampled query vectors.
        t : int
            Number of keys to select for the compacted cache.
        attention_bias : Tensor, optional
            Additive per-key attention bias for the original cache (broadcastable to (T,)).

        Returns
        -------
        C1 : Tensor, shape (t, d)
            Selected keys from K.
        beta : Tensor, shape (t,)
            Bias terms for each selected key.
        indices : list of int
            Indices of the selected keys in the original K.
        """
        """
        we are going to do this as follows:

        1. compute query gaussian params (mu is a d-dim vector, sigma is a dxd matrix)
        2. compute the linear term: K @ mu / sqrt(d)
        3. compute the quadratic term: einsum("ij,jk,ik -> i", K, sigma, K) / (2d)
        4. add, compute exponential, then top-k
        """
        T, d = K.size()
        n, _ = queries.size()
        device = K.device
        dtype_param = K.dtype

        queries32, K32 = queries.to(torch.float32), K.to(torch.float32)

        mu, sigma = queries32.mean(dim=0), queries32.T.cov()

        log_expected_unbiased = self._log_expected_exp_scores(K32, mu, sigma)
        bias32 = self._prepare_attention_bias(attention_bias, (T,))
        log_expected = log_expected_unbiased if bias32 is None else log_expected_unbiased + bias32
        key_scores = log_expected

        # Apply pooling if specified
        if self.pooling is not None:
            # key_scores is (T,), need to add batch and channel dims for pooling: (1, 1, T)
            key_scores_pooled = key_scores.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
            if self.pooling == 'avgpool':
                key_scores_pooled = F.avg_pool1d(key_scores_pooled, kernel_size=self.kernel_size,
                                                  padding=self.kernel_size // 2, stride=1)
            elif self.pooling == 'maxpool':
                key_scores_pooled = F.max_pool1d(key_scores_pooled, kernel_size=self.kernel_size,
                                                  padding=self.kernel_size // 2, stride=1)
            key_scores = key_scores_pooled.squeeze(0).squeeze(0)  # (T,) fp32

        _, selected_indices_tensor = torch.topk(key_scores, t, largest=True)
        C1 = K[selected_indices_tensor]

        if self.beta_method == 'zero':
            # Set all beta values to 0 (compute in fp32, then convert to model dtype)
            beta32 = torch.zeros(t, dtype=torch.float32, device=device)
        # The purpose of beta is to re-allocate attention mass that is lost
        # due to compaction. This branch estimates and redistributes expected
        # mass without materializing QK^T, which is required for the NNLS targets.
        elif self.beta_method == "redistribute_uniform":
            expected_mass = torch.exp(log_expected.clamp(min=-60.0, max=60.0))
            selected_mass = expected_mass[selected_indices_tensor]
            selected_base_mass = torch.exp(
                log_expected_unbiased[selected_indices_tensor].clamp(min=-60.0, max=60.0)
            )
            missing_mass = (expected_mass.sum() - selected_mass.sum()).clamp_min(0.0)
            beta32 = torch.log(
                (selected_mass + missing_mass / t).clamp_min(1e-12)
                / selected_base_mass.clamp_min(1e-12)
            )
        elif self.beta_method == "nnls":
            selected_log_expected = log_expected_unbiased[selected_indices_tensor]
            beta32 = self._compute_beta_via_expected_nnls(
                log_expected,
                selected_log_expected,
                lower_bound=self.nnls_lower_bound,
                upper_bound=self.nnls_upper_bound,
            )
        elif self.beta_method == "taylor_nnls":
            beta32 = self._compute_beta_via_taylor_nnls(
                mu,
                K32,
                C1.to(torch.float32),
                attention_bias=bias32,
                lower_bound=self.nnls_lower_bound,
                upper_bound=self.nnls_upper_bound,
            )
        else:
            raise ValueError(
                f"Unsupported beta_method '{self.beta_method}' for HighestExpectedAttentionKeysCompaction"
            )

        # Convert beta from fp32 to model dtype (e.g., bf16) for storage
        beta = beta32.to(dtype_param)

        return C1, beta, selected_indices_tensor.cpu().tolist()

    @staticmethod
    def _prepare_attention_bias(attention_bias, shape):
        if attention_bias is None:
            return None
        try:
            return torch.broadcast_to(
                attention_bias.to(torch.float32),
                shape
            )
        except Exception as e:
            raise ValueError(
                f"attention_bias must be broadcastable to {shape}, "
                f"got {tuple(attention_bias.shape)}"
            ) from e

    @staticmethod
    def _log_expected_exp_scores(keys, mu, sigma):
        """
        Computes expected attention scores given K and a query distribution
        using the Gaussian MGF.
        """
        _, d = keys.size()
        linear = keys @ mu / (d ** 0.5)
        quadratic = torch.einsum("ij,jk,ik->i", keys, sigma, keys) / (2 * d)
        return linear + quadratic

    def _compute_beta_via_expected_nnls(
        self,
        log_expected,
        selected_log_expected,
        lower_bound=None,
        upper_bound=None,
    ):
        """
        Solve for beta in the expected attention mass matching setup using
        (non-negative) least squares. Note that we have just one equation and t unknowns,
        so the system is underdetermined.
        """
        max_log = log_expected.max()
        expected_mass = torch.exp((log_expected - max_log).clamp(min=-60.0, max=60.0))
        selected_mass = torch.exp((selected_log_expected - max_log).clamp(min=-60.0, max=60.0))

        target = expected_mass.sum().unsqueeze(0)
        M = selected_mass.unsqueeze(0)
        weights = self._nnls_pg(
            M,
            target,
            self.nnls_iters,
            lower_bound,
            upper_bound,
        )
        return torch.log(weights)

    def _compute_beta_via_taylor_nnls(
        self,
        mu,
        K,
        C1,
        attention_bias=None,
        lower_bound=None,
        upper_bound=None,
    ):
        """
        Solve for beta in the expected attention mass matching setup using
        (non-negative) least squares. Rather than attempting to solve an underdetermined
        system, we generate more constraints by matching "near" mu, using the Taylor
        expansion of the attention mass formula.
        """
        _, d = K.size()
        inv_sqrt_d = (1.0 / d) ** 0.5

        bias32 = self._prepare_attention_bias(attention_bias, (K.shape[0],))

        full_scores = K @ mu * inv_sqrt_d
        if bias32 is not None:
            full_scores = full_scores + bias32
        compacted_scores = C1 @ mu * inv_sqrt_d
        ref = torch.maximum(full_scores.max(), compacted_scores.max())

        full_exp = torch.exp((full_scores - ref).clamp(min=-60.0, max=60.0))
        compacted_exp = torch.exp((compacted_scores - ref).clamp(min=-60.0, max=60.0))

        y0 = full_exp.sum().unsqueeze(0)
        y1 = (full_exp[:, None] * K * inv_sqrt_d).sum(dim=0)
        y = torch.cat((y0, y1), dim=0)

        A0 = compacted_exp.unsqueeze(0)
        A1 = (compacted_exp[:, None] * C1 * inv_sqrt_d).T
        A = torch.cat((A0, A1), dim=0)

        weights = self._nnls_pg(
            A,
            y,
            self.nnls_iters,
            lower_bound,
            upper_bound,
        )
        return torch.log(weights)

    def _compute_C2_via_expected_lsq(
        self,
        mu,
        sigma,
        K,
        V,
        C1,
        beta,
        solver,
        attention_bias=None,
        ridge_lambda=0,
        ridge_scale='spectral',
    ):
        """
        Solve for C2 (the compacted values) in the expected attention output matching
        setup using least squares. Note that we have d equations and t * d unknowns,
        so the system is underdetermined.
        """
        dtype_param = K.dtype
        K = K.to(torch.float32)
        V = V.to(torch.float32)
        C1 = C1.to(torch.float32)
        beta = beta.to(torch.float32)
        mu = mu.to(torch.float32)
        sigma = sigma.to(torch.float32)

        log_full = self._log_expected_exp_scores(K, mu, sigma)
        bias32 = self._prepare_attention_bias(attention_bias, (K.shape[0],))
        if bias32 is not None:
            log_full = log_full + bias32
        full_weights = torch.softmax(log_full, dim=0)
        Y = (full_weights @ V).unsqueeze(0)

        log_compacted = self._log_expected_exp_scores(C1, mu, sigma) + beta
        X = torch.softmax(log_compacted, dim=0).unsqueeze(0)

        debug_tensors = {
            "K": K,
            "V": V,
            "C1": C1,
            "beta": beta,
            "mu": mu,
            "sigma": sigma,
        }
        return self._solve_C2_regression(
            X,
            Y,
            dtype_param=dtype_param,
            ridge_lambda=ridge_lambda,
            solver=solver,
            ridge_scale=ridge_scale,
            debug_tensors=debug_tensors,
        )

    def _compute_C2_via_expectation_with_method(
        self,
        mu,
        K,
        V,
        C1,
        beta,
        solver,
        attention_bias=None,
        ridge_lambda=0,
        ridge_scale='spectral',
    ):
        """
        Solve for C2 (the compacted values) in the expected attention output matching
        setup using least squares. Rather than attempting to solve an underdetermined
        system, we generate more constraints by matching "near" mu, using the Taylor
        expansion of the attention output formula.

        This gives us d * (d + 1) equations with t * d unknowns, which makes the system
        well-determined/overdetermined when t <= d + 1.
        """
        dtype_param = K.dtype
        K = K.to(torch.float32)
        V = V.to(torch.float32)
        C1 = C1.to(torch.float32)
        beta = beta.to(torch.float32)
        mu = mu.to(torch.float32)

        _, d = K.size()

        inv_sqrt_d = (1.0 / d) ** 0.5

        bias32 = self._prepare_attention_bias(attention_bias, (K.shape[0],))

        scores = K @ mu * inv_sqrt_d
        if bias32 is not None:
            scores = scores + bias32
        # we want to compute K^T (diag(a)V) - (K^T a)(a^T V)
        a = torch.nn.functional.softmax(scores, dim=0)
        y_full = a @ V
        grad_full = K.T @ (a[:, None] * V) - (K.T @ a)[:, None] * (a @ V)[None, :]
        grad_full *= inv_sqrt_d

        compacted_scores = C1 @ mu
        b = torch.nn.functional.softmax(compacted_scores * inv_sqrt_d + beta, dim=0)
        grad_compacted = (C1 - (b @ C1)[None, :]).T * b[None, :]
        grad_compacted *= inv_sqrt_d

        Y = torch.cat((y_full[None, :], grad_full), dim=0) # (d + 1, d)
        X = torch.cat((b[None, :], grad_compacted), dim=0) # (d + 1, t)

        debug_tensors = {
            "K": K,
            "V": V,
            "C1": C1,
            "beta": beta,
            "mu": mu,
            "attention_bias": bias32,
        }
        return self._solve_C2_regression(
            X,
            Y,
            dtype_param=dtype_param,
            ridge_lambda=ridge_lambda,
            solver=solver,
            ridge_scale=ridge_scale,
            debug_tensors=debug_tensors,
        )
