import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        if stride != 1 or in_channels!= out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv_block(x)
        out += residual
        return self.relu(out)


def get_conv_layer(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


def get_fc_layer(in_channels, out_channels, dropout):
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.ReLU(),
        nn.Dropout(dropout)
    )


class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()
        
        self.res_block_layers = [
            {"num_blocks": 2, "out_channels": 64, "stride": 1},
            {"num_blocks": 2, "out_channels": 128 , "stride": 2},
            {"num_blocks": 2, "out_channels": 256, "stride": 2},
            {"num_blocks": 2, "out_channels": 512, "stride": 2}]
        
        self.conv_layer = get_conv_layer(3, self.res_block_layers[0]["out_channels"], kernel_size=7, stride=2, padding=3, bias=False)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.in_channels = self.res_block_layers[0]["out_channels"]
        self.res_layers = self._make_res_layers(self.res_block_layers)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.fc = get_fc_layer(self.in_channels, num_classes, dropout)

    def _make_res_layers(self, res_block_layers):
        layers = []
        for layer in res_block_layers:
            layers.append(self._make_res_layer(layer["out_channels"], layer["num_blocks"], layer["stride"]))
            self.in_channels = layer["out_channels"]
        
        return nn.Sequential(*layers)
        
    def _make_res_layer(self, out_channels, num_blocks, stride):
        layers = [ResidualBlock(self.in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.max_pool(x)
        x = self.res_layers(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
