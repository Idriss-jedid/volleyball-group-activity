import torch

from gar.models.pooling import group_pool


def test_two_groups_split_slots_0_5_and_6_11():
    x = torch.arange(12.0).view(12, 1)  # slot i has value i
    mask = torch.ones(12, dtype=torch.bool)
    assert group_pool(x, mask, groups=2).tolist() == [5.0, 11.0]
    assert group_pool(x, mask, groups=1).tolist() == [11.0]


def test_padding_is_ignored_even_with_negative_values():
    x = -torch.ones(12, 1)
    x[10:] = 100.0  # padded slots must not win the max
    mask = torch.ones(12, dtype=torch.bool)
    mask[10:] = False
    assert group_pool(x, mask, groups=2).tolist() == [-1.0, -1.0]
    assert group_pool(x, mask, groups=1, mode="avg").tolist() == [-1.0]


def test_batched_shapes():
    x, mask = torch.randn(4, 9, 12, 16), torch.ones(4, 9, 12, dtype=torch.bool)
    assert group_pool(x, mask, groups=2).shape == (4, 9, 32)
