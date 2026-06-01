import math

exp = math.exp

# Variants tested and their key-overlap / speedup tradeoffs (QuALITY, Qwen3-4B):
#   k=3, n_rough=256 → 0.797 overlap, 3.1× speedup
#   k=3, n_rough=512 → 0.811 overlap, 3.1× speedup
#   k=4, n_rough=256 → 0.852 overlap, 2.4× speedup
#   k=5, n_rough=256 → 0.890 overlap, 1.9× speedup
#   k=5, n_rough=512 → 0.900 overlap, 1.9× speedup
# See KEY_SIMILARITY.md for the full sweep.
# Direct C2 outperforms LSQ at all ratios when n_rough << t; see C2_FITTING_ANALYSIS.md.

HIGH_ATTN_BASE = {
    'algorithm': 'highest_attention_keys',
    'score_method': 'rms',
    'nnls_iters': 2,
    'nnls_lower_bound': exp(-3),
    'nnls_upper_bound': exp(3),
    'c2_solver': 'lstsq',
    'c2_ridge_lambda': 0,
    'c2_ridge_scale': 'spectral',
    'on_policy': True,
}

BASE_SELECTIVE_EXACT = {
    'algorithm': 'selective_exact',
    'on_policy': True,
    'c2_solver': 'lstsq',
    'c2_ridge_lambda': 0,
    'c2_ridge_scale': 'spectral',
    'nnls_iters': 2,
    'nnls_lower_bound': exp(-3),
    'nnls_upper_bound': exp(3),
}


def selective_exact(k, n_rough, beta_method, c2_method, score_method='rms', **overrides):
    cfg = {
        **BASE_SELECTIVE_EXACT,
        'k_factor': k,
        'n_rough': n_rough,
        'score_method': score_method,
        'beta_method': beta_method,
        'c2_method': c2_method,
    }
    cfg.update(overrides)
    return cfg


config = {
    # -------------------------------------------------------------------------
    # Baselines (highest_attention_keys algorithm)
    # -------------------------------------------------------------------------
    'high_attn': {**HIGH_ATTN_BASE, 'c2_method': 'lsq'},
    'high_attn_direct': {**HIGH_ATTN_BASE, 'c2_method': 'direct'},

    # -------------------------------------------------------------------------
    # Recommended configs: direct C2 (no LSQ overhead, best QA accuracy)
    # -------------------------------------------------------------------------
    'ser3_256_redist_direct': selective_exact(3, 256, 'redistribute_uniform', 'direct'),
    'ser3_512_redist_direct': selective_exact(3, 512, 'redistribute_uniform', 'direct'),
    'ser4_256_redist_direct': selective_exact(4, 256, 'redistribute_uniform', 'direct'),
    'ser4_512_redist_direct': selective_exact(4, 512, 'redistribute_uniform', 'direct'),
    'ser5_256_redist_direct': selective_exact(5, 256, 'redistribute_uniform', 'direct'),
    'ser5_512_redist_direct': selective_exact(5, 512, 'redistribute_uniform', 'direct'),

    'ser3_256_nnls_direct': selective_exact(3, 256, 'nnls', 'direct'),
    'ser3_512_nnls_direct': selective_exact(3, 512, 'nnls', 'direct'),
    'ser5_256_nnls_direct': selective_exact(5, 256, 'nnls', 'direct'),
    'ser5_512_nnls_direct': selective_exact(5, 512, 'nnls', 'direct'),

    # -------------------------------------------------------------------------
    # LSQ C2: worse QA accuracy than direct but included for ablations.
    # Ridge is required to stabilize the underdetermined regime (n_rough << t).
    # -------------------------------------------------------------------------
    'ser3_256_redist_lsq': selective_exact(3, 256, 'redistribute_uniform', 'lsq'),
    'ser3_256_redist_lsq_ridge1e-2': selective_exact(3, 256, 'redistribute_uniform', 'lsq', c2_ridge_lambda=1e-2),
    'ser3_512_redist_lsq_ridge1e-2': selective_exact(3, 512, 'redistribute_uniform', 'lsq', c2_ridge_lambda=1e-2),
    'ser5_256_redist_lsq_ridge1e-2': selective_exact(5, 256, 'redistribute_uniform', 'lsq', c2_ridge_lambda=1e-2),
    'ser5_512_redist_lsq_ridge1e-2': selective_exact(5, 512, 'redistribute_uniform', 'lsq', c2_ridge_lambda=1e-2),

    'ser3_256_nnls_lsq_ridge1e-2': selective_exact(3, 256, 'nnls', 'lsq', c2_ridge_lambda=1e-2),
    'ser3_512_nnls_lsq_ridge1e-2': selective_exact(3, 512, 'nnls', 'lsq', c2_ridge_lambda=1e-2),
    'ser5_256_nnls_lsq_ridge1e-2': selective_exact(5, 256, 'nnls', 'lsq', c2_ridge_lambda=1e-2),
    'ser5_512_nnls_lsq_ridge1e-2': selective_exact(5, 512, 'nnls', 'lsq', c2_ridge_lambda=1e-2),
}
