import numpy as np
import pytest
import torch

from dsac import DSAC
from policy import (action_mask, critic_action, domain_nvec, onehot_action,
                    padded_observation, quantile_huber_loss,
                    select_lower_distribution, style_value, twin_style_value)


def test_mask_and_padding():
    mask = action_mask([3, 4, 2], [2, 1, 2], can_accept=False)
    assert mask.tolist() == [[True, True, False, False], [True, False, False, False],
                             [False, True, False, False]]
    assert np.allclose(padded_observation([1, 2, 0.4], 6), [1, 2, 0, 0, 0, 0.4])
    with pytest.raises(ValueError):
        action_mask([2], [3])
    accepted_a = critic_action(onehot_action(torch.tensor([[0, 0]]), [3, 2]), [3, 2])
    accepted_b = critic_action(onehot_action(torch.tensor([[2, 0]]), [3, 2]), [3, 2])
    rejected = critic_action(onehot_action(torch.tensor([[2, 1]]), [3, 2]), [3, 2])
    assert torch.equal(accepted_a, accepted_b)
    assert not torch.equal(accepted_a, rejected)


def test_quantile_loss_and_styles():
    q = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert quantile_huber_loss(q, q).item() >= 0
    assert torch.isfinite(quantile_huber_loss(q, q)).all()
    conservative = style_value(q, "conservative", conservative_quantile=0.5).item()
    neutral = style_value(q, "neutral").item()
    aggressive = style_value(q, "aggressive", aggressive_quantile=0.5).item()
    assert conservative < neutral < aggressive
    with pytest.raises(ValueError):
        style_value(q, "unknown")
    first = torch.tensor([[1.0, 5.0, 2.0, 6.0]])
    second = torch.tensor([[3.0, 2.0, 4.0, 1.0]])
    assert select_lower_distribution(first, second).tolist() == [[1.0, 2.0, 2.0, 1.0]]
    expected = min(style_value(first).item(), style_value(second).item())
    assert twin_style_value(first, second).item() == pytest.approx(expected)


def test_dsac_update_and_action_mask():
    torch.manual_seed(5)
    model = DSAC(7, [3, 2], quantiles=8, hidden=16, batch_size=2, buffer_size=4,
                 auto_entropy=True, policy_update_frequency=1)
    obs = np.zeros(7, dtype=np.float32)
    mask = action_mask([3, 2], [2, 2], False).numpy()
    for _ in range(12):
        action = model.act(obs, mask)
        assert action[0] < 2 and action[1] == 1
    model.replay.add(obs, [0, 1], 0.5, obs, False, mask, mask)
    model.replay.add(obs, [1, 1], 0.2, obs, True, mask, mask)
    assert model.update() is not None
    assert np.isfinite(model.alpha) and model.alpha > 0
    for style in ("neutral", "aggressive", "conservative"):
        assert model.act(obs, mask, style=style, candidates=3)[1] == 1
    saved = model.state_dict()
    restored = DSAC(7, [3, 2], quantiles=8, hidden=16, batch_size=2, buffer_size=4,
                    auto_entropy=True, policy_update_frequency=1)
    restored.load_state_dict(saved)
    assert len(restored.replay) == 2
    with pytest.raises(ValueError):
        DSAC(8, [3, 2], quantiles=8, hidden=16).load_state_dict(saved)


def test_entropy_schedule():
    model = DSAC(4, [2, 2], hidden=8, quantiles=4, target_entropy_ratio=0.9,
                 target_entropy_final_ratio=0.3, entropy_anneal_steps=100)
    assert model.current_target_entropy_ratio == pytest.approx(0.9)
    model.update_steps = 50
    assert model.current_target_entropy_ratio == pytest.approx(0.6)
    model.update_steps = 200
    assert model.current_target_entropy_ratio == pytest.approx(0.3)
