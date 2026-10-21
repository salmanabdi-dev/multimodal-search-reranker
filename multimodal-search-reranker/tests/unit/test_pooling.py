import pytest

torch = pytest.importorskip("torch")

from mmsearch.embeddings.pooling import l2_normalize, mean_pool  # noqa: E402


def test_mean_pool_ignores_padding():
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.allclose(mean_pool(hidden, mask), torch.tensor([[2.0, 2.0]]))


def test_l2_normalize():
    v = l2_normalize(torch.tensor([[3.0, 4.0]]))
    assert torch.allclose(v, torch.tensor([[0.6, 0.8]]))
