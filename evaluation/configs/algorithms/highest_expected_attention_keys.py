import math

exp = math.exp

BASE_EXPECTED = {
    'algorithm': 'highest_expected_attention_keys',
    'nnls_iters': 2,
    'nnls_lower_bound': exp(-3),
    'nnls_upper_bound': exp(3),
    'c2_solver': 'lstsq',
    'c2_ridge_lambda': 0,
    'c2_ridge_scale': 'spectral',
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
}
