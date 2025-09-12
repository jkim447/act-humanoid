import numpy as np

def sample_indices(T, chunk_size, stride_f):
    # Max starting index so that the last (chunk_size-1)*stride_f stays in range
    max_start = int(np.floor(T - (chunk_size - 1) * stride_f - 1))
    if max_start < 0:
        # Not enough frames; shorten chunk to what fits
        chunk_size = max(1, int(np.floor((T - 1) / max(stride_f, 1e-6))) + 1)
        max_start = 0
    start = np.random.randint(0, max_start + 1) if max_start > 0 else 0

    idxs_f = start + np.arange(chunk_size, dtype=np.float64) * stride_f
    idxs = np.clip(np.round(idxs_f).astype(int), 0, T - 1)

    # Ensure non-decreasing indices (avoid going backwards due to rounding)
    idxs = np.maximum.accumulate(idxs)
    return idxs

# ===================== DEMO =====================
# Create dummy data: [0, 1, 2, ..., 19]
T = 20
data = np.arange(T)

examples = [
    {"chunk_size": 5, "stride": 1.0},
    {"chunk_size": 5, "stride": 1.5},
    {"chunk_size": 8, "stride": 2.0},
    {"chunk_size": 10, "stride": 2.5},
    {"chunk_size": 25, "stride": 1.5},   # longer than T (edge case)
]

for ex in examples:
    cs, st = ex["chunk_size"], ex["stride"]
    idxs = sample_indices(T, cs, st)
    sampled = data[idxs]
    print(f"chunk_size={cs}, stride={st}")
    print(" indices:", idxs)
    print(" values: ", sampled)
    print("-"*40)