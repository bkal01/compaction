import math

exp = math.exp

BASE_EXPECTED = {
    'algorithm': 'highest_expected_attention_keys',
    # Only affects methods with fit_query_subset_method set. Expected methods
    # use Gaussian expected exp-score key selection.
    'score_method': 'rms',
    'nnls_iters': 2,
    'nnls_lower_bound': exp(-3),
    'nnls_upper_bound': exp(3),
    'c2_solver': 'lstsq',
    'c2_ridge_lambda': 0,
    'c2_ridge_scale': 'spectral',
    'fit_query_subset_method': None,
    'fit_query_subset_size': None,
    'on_policy': True,
}


def expected(beta_method, c2_method, **overrides):
    cfg = {
        **BASE_EXPECTED,
        'beta_method': beta_method,
        'c2_method': c2_method,
    }
    cfg.update(overrides)
    return cfg


def sampled(beta_method, c2_method, m, score_method='rms', **overrides):
    return expected(
        beta_method,
        c2_method,
        score_method=score_method,
        fit_query_subset_method='sampled',
        fit_query_subset_size=m,
        **overrides,
    )


def synthetic(beta_method, c2_method, m, score_method='rms', **overrides):
    return expected(
        beta_method,
        c2_method,
        score_method=score_method,
        fit_query_subset_method='synthetic',
        fit_query_subset_size=m,
        **overrides,
    )


def mahalanobis(beta_method, c2_method, m, score_method='rms', **overrides):
    return expected(
        beta_method,
        c2_method,
        score_method=score_method,
        fit_query_subset_method='mahalanobis_sample',
        fit_query_subset_size=m,
        **overrides,
    )


def pc_extreme(beta_method, c2_method, m, score_method='rms', **overrides):
    return expected(
        beta_method,
        c2_method,
        score_method=score_method,
        fit_query_subset_method='pc_extreme_sample',
        fit_query_subset_size=m,
        **overrides,
    )


config = {
    # Main full-attention baseline. This uses the same C2 solver default as the
    # expected-attention methods below.
    'high_attn': {
        'algorithm': 'highest_attention_keys',
        'score_method': 'rms',
        'nnls_iters': 2,
        'nnls_lower_bound': exp(-3),
        'nnls_upper_bound': exp(3),
        'c2_method': 'lsq',
        'c2_solver': 'lstsq',
        'c2_ridge_lambda': 0,
        'c2_ridge_scale': 'spectral',
        'on_policy': True,
    },
    'high_attn_ridge1e-2': {
        'algorithm': 'highest_attention_keys',
        'score_method': 'rms',
        'nnls_iters': 2,
        'nnls_lower_bound': exp(-3),
        'nnls_upper_bound': exp(3),
        'c2_method': 'lsq',
        'c2_solver': 'lstsq',
        'c2_ridge_lambda': 1e-2,
        'c2_ridge_scale': 'spectral',
        'on_policy': True,
    },

    # Full beta_method x c2_method grid.
    'expected_zero_direct': expected('zero', 'direct'),
    'expected_zero_lsq': expected('zero', 'lsq'),
    'expected_zero_taylor': expected('zero', 'taylor_lsq'),

    'expected_redist_direct': expected('redistribute_uniform', 'direct'),
    'expected_redist_lsq': expected('redistribute_uniform', 'lsq'),
    'expected_redist_taylor': expected('redistribute_uniform', 'taylor_lsq'),

    'expected_nnls_direct': expected('nnls', 'direct'),
    'expected_nnls_lsq': expected('nnls', 'lsq'),
    'expected_nnls_taylor': expected('nnls', 'taylor_lsq'),

    'expected_taylorbeta_direct': expected('taylor_nnls', 'direct'),
    'expected_taylorbeta_lsq': expected('taylor_nnls', 'lsq'),
    'expected_taylorbeta_taylor': expected('taylor_nnls', 'taylor_lsq'),

    # Ridge variants for fitted C2 paths. Use these to compare norm
    # regularization against Taylor/Jacobian augmentation in underdetermined
    # regimes.
    'expected_redist_lsq_ridge1e-4': expected(
        'redistribute_uniform', 'lsq', c2_ridge_lambda=1e-4
    ),
    'expected_redist_lsq_ridge1e-3': expected(
        'redistribute_uniform', 'lsq', c2_ridge_lambda=1e-3
    ),
    'expected_redist_lsq_ridge1e-2': expected(
        'redistribute_uniform', 'lsq', c2_ridge_lambda=1e-2
    ),

    'expected_zero_taylor_ridge1e-4': expected(
        'zero', 'taylor_lsq', c2_ridge_lambda=1e-4
    ),
    'expected_zero_taylor_ridge1e-2': expected(
        'zero', 'taylor_lsq', c2_ridge_lambda=1e-2
    ),
    'expected_redist_taylor_ridge1e-4': expected(
        'redistribute_uniform', 'taylor_lsq', c2_ridge_lambda=1e-4
    ),
    'expected_redist_taylor_ridge1e-3': expected(
        'redistribute_uniform', 'taylor_lsq', c2_ridge_lambda=1e-3
    ),
    'expected_redist_taylor_ridge1e-2': expected(
        'redistribute_uniform', 'taylor_lsq', c2_ridge_lambda=1e-2
    ),
    'expected_nnls_taylor_ridge1e-4': expected(
        'nnls', 'taylor_lsq', c2_ridge_lambda=1e-4
    ),
    'expected_nnls_taylor_ridge1e-2': expected(
        'nnls', 'taylor_lsq', c2_ridge_lambda=1e-2
    ),
    'expected_taylorbeta_taylor_ridge1e-4': expected(
        'taylor_nnls', 'taylor_lsq', c2_ridge_lambda=1e-4
    ),
    'expected_taylorbeta_taylor_ridge1e-2': expected(
        'taylor_nnls', 'taylor_lsq', c2_ridge_lambda=1e-2
    ),

    # Sampled-query key selection + beta/C2 fitting variants. The sampled
    # queries are used like ordinary attention queries: score keys by
    # mean/max/rms attention mass, then fit beta/C2 against the same
    # sampled-query attention constraints.
    'expected_redist_lsq_sampled64_ridge1e-2': sampled(
        'redistribute_uniform', 'lsq', 64, c2_ridge_lambda=1e-2
    ),
    'expected_redist_lsq_sampled128_ridge1e-2': sampled(
        'redistribute_uniform', 'lsq', 128, c2_ridge_lambda=1e-2
    ),
    'expected_redist_lsq_sampled256_ridge1e-2': sampled(
        'redistribute_uniform', 'lsq', 256, c2_ridge_lambda=1e-2
    ),
    'expected_redist_lsq_sampled256_mean_ridge1e-2': sampled(
        'redistribute_uniform', 'lsq', 256, score_method='mean', c2_ridge_lambda=1e-2
    ),
    'expected_redist_lsq_sampled256_max_ridge1e-2': sampled(
        'redistribute_uniform', 'lsq', 256, score_method='max', c2_ridge_lambda=1e-2
    ),
    'expected_redist_lsq_sampled256_rms_ridge1e-2': sampled(
        'redistribute_uniform', 'lsq', 256, score_method='rms', c2_ridge_lambda=1e-2
    ),
    'expected_redist_lsq_sampled512_ridge1e-2': sampled(
        'redistribute_uniform', 'lsq', 512, c2_ridge_lambda=1e-2
    ),
    'expected_nnls_lsq_sampled64_ridge1e-2': sampled(
        'nnls', 'lsq', 64, c2_ridge_lambda=1e-2
    ),
    'expected_nnls_lsq_sampled128_ridge1e-2': sampled(
        'nnls', 'lsq', 128, c2_ridge_lambda=1e-2
    ),
    'expected_nnls_lsq_sampled256_ridge1e-2': sampled(
        'nnls', 'lsq', 256, c2_ridge_lambda=1e-2
    ),
    'expected_nnls_lsq_sampled512_ridge1e-2': sampled(
        'nnls', 'lsq', 512, c2_ridge_lambda=1e-2
    ),
    'expected_nnls_direct_sampled256': sampled(
        'nnls', 'direct', 256
    ),
    'expected_zero_direct_sampled256': sampled(
        'zero', 'direct', 256
    ),
    'expected_redist_direct_sampled256': sampled(
        'redistribute_uniform', 'direct', 256
    ),
    'expected_redist_lsq_sampled256': sampled(
        'redistribute_uniform', 'lsq', 256
    ),
    'expected_nnls_lsq_sampled256': sampled(
        'nnls', 'lsq', 256
    ),

    # Mahalanobis-query key selection + beta/C2 fitting variants. These use the
    # real queries furthest from the empirical Gaussian fit, then follow the same
    # attention scoring and ridge LSQ path.
    'expected_redist_lsq_mahalanobis256_ridge1e-2': mahalanobis(
        'redistribute_uniform', 'lsq', 256, c2_ridge_lambda=1e-2
    ),
    'expected_redist_lsq_mahalanobis256_ridge1e-1': mahalanobis(
        'redistribute_uniform', 'lsq', 256, c2_ridge_lambda=1e-1
    ),
    'expected_redist_lsq_mahalanobis1024_ridge1e-2': mahalanobis(
        'redistribute_uniform', 'lsq', 1024, c2_ridge_lambda=1e-2
    ),
    'expected_redist_direct_mahalanobis256': mahalanobis(
        'redistribute_uniform', 'direct', 256
    ),
    'expected_zero_direct_mahalanobis256': mahalanobis(
        'zero', 'direct', 256
    ),
    'expected_nnls_lsq_mahalanobis256_ridge1e-2': mahalanobis(
        'nnls', 'lsq', 256, c2_ridge_lambda=1e-2
    ),
    'expected_nnls_lsq_mahalanobis1024_ridge1e-2': mahalanobis(
        'nnls', 'lsq', 1024, c2_ridge_lambda=1e-2
    ),
    'expected_nnls_direct_mahalanobis256': mahalanobis(
        'nnls', 'direct', 256
    ),
    'expected_zero_direct_pc_extreme384': pc_extreme(
        'zero', 'direct', 384
    ),
    'expected_zero_direct_pc_extreme448': pc_extreme(
        'zero', 'direct', 448
    ),
    'expected_redist_direct_pc_extreme384': pc_extreme(
        'redistribute_uniform', 'direct', 384
    ),
    'expected_redist_direct_pc_extreme448': pc_extreme(
        'redistribute_uniform', 'direct', 448
    ),
    'expected_redist_lsq_pc_extreme384_ridge1e-2': pc_extreme(
        'redistribute_uniform', 'lsq', 384, c2_ridge_lambda=1e-2
    ),
    'expected_redist_lsq_pc_extreme448_ridge1e-2': pc_extreme(
        'redistribute_uniform', 'lsq', 448, c2_ridge_lambda=1e-2
    ),

    # Synthetic-query key selection + beta/C2 fitting variants. These use
    # deterministic Gaussian sigma-point queries instead of sampled real
    # queries, then follow the same attention scoring and ridge LSQ path.
    'expected_redist_lsq_synthetic64_ridge1e-2': synthetic(
        'redistribute_uniform', 'lsq', 64, c2_ridge_lambda=1e-2
    ),
    'expected_redist_lsq_synthetic129_ridge1e-2': synthetic(
        'redistribute_uniform', 'lsq', 129, c2_ridge_lambda=1e-2
    ),
    'expected_redist_lsq_synthetic257_ridge1e-2': synthetic(
        'redistribute_uniform', 'lsq', 257, c2_ridge_lambda=1e-2
    ),

    # Diagnostic variants for inspecting C2 regression conditioning/residuals.
    # These add per-layer/head JSON stats and should be used in small runs.
    'expected_zero_lsq_diag': expected('zero', 'lsq', c2_diagnostics=True),
    'expected_zero_taylor_diag': expected('zero', 'taylor_lsq', c2_diagnostics=True),
    'expected_redist_lsq_diag': expected('redistribute_uniform', 'lsq', c2_diagnostics=True),
    'expected_redist_taylor_diag': expected('redistribute_uniform', 'taylor_lsq', c2_diagnostics=True),
    'expected_zero_taylor_ridge1e-2_diag': expected(
        'zero', 'taylor_lsq', c2_ridge_lambda=1e-2, c2_diagnostics=True
    ),
    'expected_redist_taylor_ridge1e-2_diag': expected(
        'redistribute_uniform', 'taylor_lsq', c2_ridge_lambda=1e-2, c2_diagnostics=True
    ),
    'expected_redist_lsq_sampled256_ridge1e-2_diag': sampled(
        'redistribute_uniform', 'lsq', 256, c2_ridge_lambda=1e-2, c2_diagnostics=True
    ),
    'expected_redist_lsq_mahalanobis256_ridge1e-2_diag': mahalanobis(
        'redistribute_uniform', 'lsq', 256, c2_ridge_lambda=1e-2, c2_diagnostics=True
    ),
    'expected_nnls_lsq_sampled256_ridge1e-2_diag': sampled(
        'nnls', 'lsq', 256, c2_ridge_lambda=1e-2, c2_diagnostics=True
    ),
    'expected_nnls_lsq_mahalanobis256_ridge1e-2_diag': mahalanobis(
        'nnls', 'lsq', 256, c2_ridge_lambda=1e-2, c2_diagnostics=True
    ),
}
