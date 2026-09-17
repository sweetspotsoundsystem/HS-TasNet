"""A fixed, memoryless correction using all four native raw source estimates.

The working policy assigns the whole raw mixture discrepancy to Other.
This diagnostic instead assigns 1/16 to each of Drums, Bass and Vocals,
leaving 13/16 for Other. It changes no weights or source gains. It has not
been integrated into an inference model, export or plugin.
"""
from __future__ import annotations

VERSION = "latency58-fixed-raw-residual-share-sixteenth-v1"
PRIMARY_SHARE = 1 / 16


def residual_share(raw, mixture, *, share=PRIMARY_SHARE):
    """Return FP32 DBV + share*(mixture-sum(raw4)), then residual Other.

Inputs are native FP32 [4,2,T] raw estimates and physical [2,T] mixture.
The zero-share branch reproduces the existing shipping policy exactly.
"""
    import numpy as np
    from research.direct.evaluate import shipping_residual

    if (not isinstance(raw, np.ndarray) or not isinstance(mixture, np.ndarray)
            or raw.ndim != 3 or raw.shape[:2] != (4, 2) or raw.shape[-1] < 1
            or mixture.shape != raw.shape[1:] or raw.dtype != np.float32
            or mixture.dtype != np.float32 or not np.isfinite(raw).all()
            or not np.isfinite(mixture).all() or type(share) not in (int, float)
            or share not in (0, PRIMARY_SHARE)):
        raise ValueError("Require finite native FP32 raw4/mixture and fixed share 0 or 1/16")
    if not share:
        return shipping_residual(raw, mixture)
    discrepancy = mixture - raw.sum(axis=0, dtype=np.float32)
    corrected = raw.copy()
    corrected[:3] += np.float32(share) * discrepancy[None]
    return shipping_residual(corrected, mixture)
