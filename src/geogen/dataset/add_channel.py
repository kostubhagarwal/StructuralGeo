import numpy as np
from scipy.ndimage import gaussian_filter

# Example: Map density (g/cc, std_dev) to 'Rock Category Mapping' labels.
DENSITY_LU_TABLE = {
    0: 2.70,                                                         # bedrock
    1: (2.10, 0.05), 2: (2.20, 0.05), 3: (2.25, 0.05), 4: (2.30, 0.05), 5: (2.35, 0.05),
    6: (2.85, 0.03), 7: (2.90, 0.03), 8: (2.95, 0.03),               # dikes
    9: (2.65, 0.03), 10: (2.70, 0.03), 11: (2.75, 0.03),             # intrusions
    12: (3.40, 0.10), 13: (3.80, 0.15),                              # blobs/ores
}

def add_channel(labels, table=DENSITY_LU_TABLE, nan_lu=0.0, gaussian_sigma=None, seed=None):
    """
    Map channel to 'Rock Category Mapping' label.
    """
    rng = np.random.default_rng(seed)
    channel = np.zeros_like(labels, dtype=float)
    nan_idx = np.isnan(labels)
    channel[nan_idx] = nan_lu
    lab = labels.copy()
    lab[nan_idx] = -999  # sentinel

    for k, spec in table.items():
        m = (lab == k)
        if not np.any(m):
            continue
        if np.isscalar(spec):
            vals = np.full(m.sum(), spec, dtype=float)
        else:
            mu, sd = spec
            vals = rng.normal(mu, sd, size=m.sum())
        channel[m] = vals

    if gaussian_sigma and gaussian_sigma > 0:
        sm = gaussian_filter(channel, gaussian_sigma)
        channel = np.where(nan_idx, channel, sm)

    return channel
    