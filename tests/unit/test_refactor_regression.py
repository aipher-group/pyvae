import pytest

# The refactor into shared components must not change behavior. Pin it by
# training a tiny model for a few epochs with a fixed seed and asserting the
# loss matches a value recorded BEFORE the refactor.
#
# Workflow:
#   1. Before refactoring, run a short fixed-seed training and copy the final
#      loss into EXPECTED_FINAL_LOSS below.
#   2. Do the refactor.
#   3. Remove the skip and run this test; it must match.


EXPECTED_FINAL_LOSS = None  # fill in from a pre-refactor fixed-seed run


@pytest.mark.skip(reason="enable once the refactor is in place")
def test_refactor_preserves_loss():
    """Same seed and data give the same loss after the component refactor.

    Build a small InformedVAE, train a few epochs with a fixed seed on
    deterministic random data, and assert the final loss equals
    EXPECTED_FINAL_LOSS within a tight tolerance (e.g. 1e-5).
    """
    raise NotImplementedError
