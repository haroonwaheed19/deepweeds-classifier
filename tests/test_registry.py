from torch import nn

from weedcls.models.registry import (
    build_model,
    freeze_for_finetune,
    head_parameters,
    set_train_mode,
)


def _mobilenet():
    return build_model({"name": "mobilenet_v3_large", "pretrained": False}, 9)


def test_head_is_9_way_and_small_cnn_builds():
    m = _mobilenet()
    assert m.classifier[-1].out_features == 9
    assert build_model({"name": "small_cnn"}, 9).head[-1].out_features == 9


def test_freeze_last_n_blocks_only():
    m = _mobilenet()
    freeze_for_finetune(m, "mobilenet_v3_large", 2)
    blocks = list(m.features.children())
    assert len(blocks) == 17
    flags = [all(p.requires_grad for p in b.parameters()) for b in blocks]
    assert flags[-2:] == [True, True] and not any(flags[:-2])
    assert all(p.requires_grad for p in head_parameters(m, "mobilenet_v3_large"))


def test_frozen_batchnorm_stays_in_eval_mode():
    m = _mobilenet()
    freeze_for_finetune(m, "mobilenet_v3_large", 1)
    set_train_mode(m)
    first_bn = next(x for x in m.features[0].modules() if isinstance(x, nn.BatchNorm2d))
    last_bn = next(x for x in m.features[-1].modules() if isinstance(x, nn.BatchNorm2d))
    assert not first_bn.training and last_bn.training
