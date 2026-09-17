"""Average four independent B4 loss gradients before one Adam update."""
from research.direct.latency58_checkpoint import require

VERSION = "latency58-mean-four-b4-gradients-v1"
MICROBATCH_SIZE = 4
ACCUMULATION_STEPS = 4


def backward_mean_loss(loss):
    """Accumulate one quarter of a microbatch loss; never clear or clip gradients.

    The caller clears gradients once before four calls and clips once afterward.
    The objective is the mean of four original B4 objectives. Their capped
    projection penalties remain local to B4; this is not a claim of equivalence
    to computing that nonlinear penalty on a single B16 forward pass.
    """
    require(loss.ndim == 0 and loss.requires_grad, "Expected a live scalar microbatch loss")
    (loss / ACCUMULATION_STEPS).backward()
