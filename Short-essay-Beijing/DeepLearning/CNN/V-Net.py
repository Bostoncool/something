import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation注意力模块"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 如果输入输出通道数不同，使用1x1卷积进行调整
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        
        self.se = SEBlock(out_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(residual)
        out = F.relu(out)
        return out

class DenseBlock(nn.Module):
    """密集连接块"""
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                ResidualBlock(in_channels + i * growth_rate, growth_rate)
            )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return torch.cat(features, 1)

class AttentionBlock(nn.Module):
    """改进的注意力模块"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(F_int)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.se(psi)
        psi = self.psi(psi)
        return x * psi

class UNetPlusPlus(nn.Module):
    """增强版U-Net++网络实现，包含深度监督、注意力机制、残差连接和密集连接"""
    def __init__(self, in_channels=1, out_channels=1, 
                 features=[32, 64, 128, 256, 512, 1024], 
                 deep_supervision=True,
                 growth_rate=32,
                 num_layers=4):
        super(UNetPlusPlus, self).__init__()
        self.deep_supervision = deep_supervision
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attention_blocks = nn.ModuleList()
        
        # 初始卷积层
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True)
        )
        
        # 下采样路径
        for i in range(len(features)-1):
            self.downs.append(
                DenseBlock(features[i], growth_rate, num_layers)
            )
            self.downs.append(
                nn.Conv2d(features[i] + growth_rate * num_layers, features[i+1], kernel_size=1)
            )

        # 瓶颈层
        self.bottleneck = DenseBlock(features[-1], growth_rate, num_layers)

        # 上采样路径和注意力模块
        for i, feature in enumerate(reversed(features[:-1])):
            self.ups.append(
                nn.ConvTranspose2d(
                    features[-1-i] + growth_rate * num_layers, 
                    feature, 
                    kernel_size=2, 
                    stride=2
                )
            )
            self.ups.append(
                DenseBlock(feature * 2, growth_rate, num_layers)
            )
            if i < len(features)-2:  # 除了最后一层，都添加注意力模块
                self.attention_blocks.append(
                    AttentionBlock(feature, feature, feature//2)
                )

        # 深度监督输出层
        self.deep_supervision_conv = nn.ModuleList([
            nn.Conv2d(features[0] + growth_rate * num_layers, out_channels, kernel_size=1)
            for _ in range(len(features)-1)
        ])
        
        # 最终输出层
        self.final_conv = nn.Conv2d(features[0] + growth_rate * num_layers, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        deep_supervision_outputs = []

        # 初始特征提取
        x = self.initial(x)

        # 下采样路径
        for i in range(0, len(self.downs), 2):
            x = self.downs[i](x)
            x = self.downs[i+1](x)
            skip_connections.append(x)
            x = self.pool(x)

        # 瓶颈层
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # 上采样路径
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = F.resize(x, size=skip_connection.shape[2:])

            # 应用注意力机制
            if idx//2 < len(self.attention_blocks):
                skip_connection = self.attention_blocks[idx//2](x, skip_connection)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

            # 深度监督输出
            if self.deep_supervision and idx//2 < len(self.deep_supervision_conv):
                deep_supervision_outputs.append(self.deep_supervision_conv[idx//2](x))

        if self.deep_supervision:
            return self.final_conv(x), deep_supervision_outputs
        return self.final_conv(x)

class DiceLoss(nn.Module):
    """Dice损失函数"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - ((2. * intersection + self.smooth) / 
                    (pred_flat.sum() + target_flat.sum() + self.smooth))

class FocalLoss(nn.Module):
    """Focal损失函数"""
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    """组合损失函数：Dice Loss + Focal Loss"""
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        
    def forward(self, pred, target):
        if isinstance(pred, tuple):  # 处理深度监督的情况
            main_pred, deep_sup_preds = pred
            loss = self.alpha * self.dice_loss(main_pred, target) + \
                   (1 - self.alpha) * self.focal_loss(main_pred, target)
            
            # 添加深度监督损失
            for deep_pred in deep_sup_preds:
                loss += 0.4 * (self.alpha * self.dice_loss(deep_pred, target) + \
                             (1 - self.alpha) * self.focal_loss(deep_pred, target))
            return loss
        else:
            return self.alpha * self.dice_loss(pred, target) + \
                   (1 - self.alpha) * self.focal_loss(pred, target)

def test():
    """测试函数"""
    x = torch.randn((3, 1, 161, 161))
    model = UNetPlusPlus(
        in_channels=1, 
        out_channels=1, 
        deep_supervision=True,
        growth_rate=32,
        num_layers=4
    )
    preds = model(x)
    
    if isinstance(preds, tuple):
        main_pred, deep_sup_preds = preds
        print(f"主输出形状: {main_pred.shape}")
        print(f"深度监督输出数量: {len(deep_sup_preds)}")
        for i, pred in enumerate(deep_sup_preds):
            print(f"深度监督输出 {i+1} 形状: {pred.shape}")
    else:
        print(f"输出形状: {preds.shape}")
    
    print(f"输入形状: {x.shape}")
    print("测试成功！")

if __name__ == "__main__":
    test()

