import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class ResidualBlock(BaseModel):
    def __init__(self, in_channels, out_channels=None, reduce_grid=False):
        super().__init__()
        if out_channels:
            self.increase_dim = True
        else:
            self.increase_dim = False
            out_channels = in_channels

        stride = 2 if reduce_grid else 1
        
        self.residual_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        if self.increase_dim:
            self.conv1_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
    
    def forward(self, x):
        residual = self.residual_layer(x)
        if self.increase_dim:
            x = self.conv1_layer(x)
        return F.relu(residual + x)

class ResNet32Model(BaseModel):
    def __init__(self, base_channels, num_classes=10):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, 1, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
        )
        self.residual_layers = nn.Sequential(
            ResidualBlock(base_channels),
            ResidualBlock(base_channels),
            ResidualBlock(base_channels),
            ResidualBlock(base_channels),
            ResidualBlock(base_channels),
            ResidualBlock(base_channels, base_channels*2, True),
            ResidualBlock(base_channels*2),
            ResidualBlock(base_channels*2),
            ResidualBlock(base_channels*2),
            ResidualBlock(base_channels*2),
            ResidualBlock(base_channels*2, base_channels*4, True),
            ResidualBlock(base_channels*4),
            ResidualBlock(base_channels*4),
            ResidualBlock(base_channels*4),
            ResidualBlock(base_channels*4),
            ResidualBlock(base_channels*4, base_channels*8, True),
            ResidualBlock(base_channels*8),
            ResidualBlock(base_channels*8),
            ResidualBlock(base_channels*8),
            ResidualBlock(base_channels*8),
        )
        self.global_avg = nn.AvgPool2d(4)
        self.fc = nn.Linear(base_channels*8, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                            nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        [TODO] 논문의 Figure 3 형태로 forward를 구성해주세요
            - 단, 논문의 구현과는 다른 형태이기 때문에 컨셉만 맞추어 구성하시면 됩니다.
            - layer1, residual_layers, global_avg, fc를 조합해주세요.
        """
        return x        

