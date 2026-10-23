"""name -> model, plus the freezing/param-group logic used for fine-tuning."""
from torch import nn

from weedcls.models.backbones import build_backbone
from weedcls.models.small_cnn import SmallCNN


def _blocks(model: nn.Module, name: str) -> list[nn.Module]:
    """Backbone stages in input->output order (the unit we unfreeze by)."""
    if name in ("mobilenet_v3_large", "efficientnet_b0"):
        return list(model.features.children())
    if name == "resnet18":
        return [nn.Sequential(model.conv1, model.bn1), model.layer1, model.layer2,
                model.layer3, model.layer4]
    raise ValueError(name)


def build_model(cfg: dict, num_classes: int = 9) -> nn.Module:
    name = cfg["name"]
    if name == "small_cnn":
        return SmallCNN(num_classes, dropout=cfg.get("dropout", 0.3))
    model = build_backbone(name, pretrained=cfg.get("pretrained", True))
    if name == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:  # torchvision mobilenet/efficientnet: last classifier layer is the 1000-way head
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def head_parameters(model: nn.Module, name: str) -> list[nn.Parameter]:
    head = model.fc if name == "resnet18" else model.classifier
    return list(head.parameters())


def freeze_for_finetune(model: nn.Module, name: str, unfreeze_last: int) -> None:
    """Freeze every backbone stage except the last `unfreeze_last`; the head always trains."""
    if name == "small_cnn":
        return
    blocks = _blocks(model, name)
    for i, b in enumerate(blocks):
        trainable = i >= len(blocks) - unfreeze_last
        for p in b.parameters():
            p.requires_grad = trainable
    for p in head_parameters(model, name):
        p.requires_grad = True


def set_train_mode(model: nn.Module) -> None:
    """train() everywhere, then put fully-frozen modules back in eval() so their BatchNorm
    running stats are not silently overwritten by this dataset's statistics."""
    model.train()
    for m in model.modules():
        params = list(m.parameters(recurse=False))
        if params and not any(p.requires_grad for p in params):
            m.eval()
