import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ======================
# 1. 定义超小U-Net模型（CPU能跑）
# ======================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNetSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = DoubleConv(1, 8)
        self.d2 = DoubleConv(8, 16)
        self.up1 = DoubleConv(16 + 8, 8)
        self.out = nn.Conv2d(8, 1, 1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = F.max_pool2d(x1, 2)
        x2 = self.d2(x2)
        x2 = F.interpolate(x2, scale_factor=2)
        x = torch.cat([x2, x1], dim=1)
        x = self.up1(x)
        return torch.sigmoid(self.out(x))


# ======================
# 2. 定义Dice损失（医学分割必备）
# ======================
def dice_loss(pred, mask):
    inter = (pred * mask).sum()
    union = pred.sum() + mask.sum()
    return 1 - (2. * inter + 1) / (union + 1)


def dice_score(pred, mask):
    pred = (pred > 0.5).float()
    inter = (pred * mask).sum()
    union = pred.sum() + mask.sum()
    return (2. * inter + 1) / (union + 1)


# ======================
# 3. 生成虚拟数据（不用下载数据集！）
# ======================
print("开始生成虚拟图像...")
img = torch.randn(1, 1, 64, 64)  # 小图，CPU飞快
mask = torch.randint(0, 2, (1, 1, 64, 64)).float()

# ======================
# 4. 初始化模型
# ======================
model = UNetSmall()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ======================
# 5. 开始训练（只跑20轮，看流程）
# ======================
print("开始训练...\n")

for epoch in range(20):
    optimizer.zero_grad()
    pred = model(img)
    loss = dice_loss(pred, mask)
    loss.backward()
    optimizer.step()
    dice = dice_score(pred, mask)
    print(f"轮次 {epoch + 1} | 损失: {loss:.4f} | Dice: {dice:.4f}")

print("\n✅ 流程跑完啦！")
print("损失越来越小 = 模型在学习")
print("Dice越来越高 = 分割效果越来越好")
