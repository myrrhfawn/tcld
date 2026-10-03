import pytest
import torch


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def dfine_s_random():
    """D-FINE-S with random weights (no checkpoint, no backbone download). Enough to
    test shapes, modes and the corners->boxes identity on CPU."""
    from tcld.model import load_dfine

    torch.manual_seed(0)
    return load_dfine("dfine_s", checkpoint=None, device="cpu", strict_checkpoint=False)
