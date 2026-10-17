import torch

from gar.models.b4 import B4
from gar.models.hier import Hierarchical
from gar.models.person_lstm import PersonLSTM


def test_hierarchical_variants_shapes():
    x, mask = torch.randn(3, 9, 12, 32), torch.ones(3, 12, dtype=torch.bool)
    mask[0, 10:] = False
    for person in (None, PersonLSTM(hidden=8, dropout=0.0, dim=32)):
        for scene in (False, True):
            for groups in (1, 2):
                m = Hierarchical(person, scene, groups, "max", fc_dim=16, scene_hidden=8, dropout=0.0, dim=32)
                assert m(x, mask).shape == (3, 8)


def test_frozen_person_lstm_gets_no_gradient():
    person = PersonLSTM(hidden=8, dropout=0.0, dim=32)
    for p in person.parameters():
        p.requires_grad = False
    m = Hierarchical(person, True, 2, "max", fc_dim=16, scene_hidden=8, dropout=0.1, dim=32).train()
    assert not person.training
    m(torch.randn(2, 9, 12, 32), torch.ones(2, 12, dtype=torch.bool)).sum().backward()
    assert all(p.grad is None for p in person.parameters())


def test_b4_shape():
    assert B4(hidden=8, dropout=0.0, dim=32)(torch.randn(2, 9, 32)).shape == (2, 8)


def test_actor_transformer_shapes_and_padding():
    from gar.models.transformer import ActorTransformer

    m = ActorTransformer(d_model=16, layers=1, heads=2, dropout=0.0, dim=32).eval()
    x, boxes = torch.randn(2, 9, 12, 32), torch.rand(2, 9, 12, 4)
    mask = torch.ones(2, 12, dtype=torch.bool)
    mask[1, 10:] = False
    out = m(x, mask, boxes)
    assert out.shape == (2, 8)
    x2 = x.clone()
    x2[1, :, 10:] = 1e3  # padded players must not change the output
    assert torch.allclose(m(x2, mask, boxes)[1], out[1], atol=1e-5)
