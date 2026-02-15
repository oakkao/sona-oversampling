from scipy.spatial.distance import cdist
import numpy as np

def SONA(X, y, min_label, new_label=0):
    X_gen_min = X[y == min_label]
    X_gen_maj = X[y != min_label]

    maj_size = len(X_gen_maj)
    minor_size = len(X_gen_min)
    synthese_len = maj_size - minor_size

    if synthese_len <= 0:
        return X, y

    # 1. Distance Matrices
    dist_min2maj = cdist(X_gen_min, X_gen_maj)
    
    # 2. Identify Borders
    closest_min_idx = np.argmin(dist_min2maj, axis=0)
    neg_border = np.bincount(closest_min_idx, minlength=minor_size)

    # Find index of closest majority for each minority (Positive Border)
    closest_maj_idx = np.argmin(dist_min2maj, axis=1)
    pos_border = np.bincount(closest_maj_idx, minlength=maj_size)


    # 3. Calculate Radius
    safe_maj_mask = (pos_border == 0)
    if not np.any(safe_maj_mask):
        neg_radius = np.min(dist_min2maj, axis=1)
    else:
        neg_radius = np.min(dist_min2maj[:, safe_maj_mask], axis=1)

    # 4. Sampling Probabilities
    prop_min = 1.0 / (neg_border + 1.0)
    prop_min /= prop_min.sum()

    # 5. Generate Synthetic Samples
    idx_i = np.random.choice(minor_size, size=synthese_len, p=prop_min)
    X_i = X_gen_min[idx_i]
    
    # Pick neighbor points (j) for each i
    dist_min2min = cdist(X_gen_min, X_gen_min)
    
    # Apply inverse distance weighting for neighbors
    with np.errstate(divide='ignore'):
        terminal_weights = 1.0 / dist_min2min
    np.fill_diagonal(terminal_weights, 0) # Can't pick itself
    terminal_weights = np.nan_to_num(terminal_weights, posinf=0)
    
    # Normalize each row to be a probability distribution
    row_sums = terminal_weights.sum(axis=1, keepdims=True)
    terminal_probs = terminal_weights / row_sums

    # For each selected i, pick a neighbor j
    idx_j = [np.random.choice(minor_size, p=terminal_probs[i]) for i in idx_i]
    X_j = X_gen_min[idx_j]

    # 6. Direction and Interpolation
    diff = X_j - X_i
    norm_v = np.linalg.norm(diff, axis=1, keepdims=True)
    # Avoid division by zero for identical points
    norm_v_safe = np.where(norm_v == 0, 1, norm_v)
    
    direction_vector = diff / norm_v_safe
    alpha = np.random.random((synthese_len, 1))
    
    # Determine step size: min(radius of point i, distance to point j)
    step_size = np.minimum(neg_radius[idx_i].reshape(-1, 1), norm_v)
    
    syn_samples = X_i + (alpha * direction_vector * step_size)

    return (
        np.vstack([X, syn_samples]),
        np.hstack([y, np.repeat(min_label + new_label, synthese_len)]),
    )