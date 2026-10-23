import itertools

from torch import nn


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """Four conv blocks + GAP. Deliberately tiny so it trains on 2 CPU cores."""

    def __init__(self, num_classes: int = 9, widths=(32, 64, 128, 192), dropout: float = 0.3):
        super().__init__()
        chans = (3, *widths)
        self.features = nn.Sequential(*[_block(a, b) for a, b in itertools.pairwise(chans)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(widths[-1], num_classes))

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.head(x)
