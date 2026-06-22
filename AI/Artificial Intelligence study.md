
**整体框架**
```
人工智能AI

├─ 机器学习（ML）
│ ├─ 传统机器学习（SVM、随机森林、逻辑回归等）
│	├─ 监督：逻辑回归、SVM、随机森林、XGBoost；
│    ├─ 非监督：K-Means、DBSCAN、PCA、t-SNE、层次聚类。    
│ └─ 深度学习（Deep Learning）（基于神经网络的分支）
│	├─监督：分类 / 回归网络、ResNet、目标检测、分割；
│	├─非监督/自监督：对比学习 (SimCLR/MoCo)、聚类网络
│
│ ├─ 计算机视觉（CV）
│
│ ├─ 自然语言处理（NLP）
│
│ ├─ 语音识别
│
│ ├─ 强化学习（RL）
│
│ └─ 多模态（CV+NLP 等融合方向）
│
└─ 其他分支（符号主义、专家系统等，现在主流应用较少）

```
**学习人工智能首先要认可以下基本理念**
Learning artificial intelligence must first recognize the following basic concepts.
+ **Functions describe the world！**
	+ 牛顿第二定律: Newton's Second Law of Motion$$F = ma$$
	+ 万有引力定律: Law of Universal Gravitation$$F = G \frac{m_1 m_2}{r^2}$$
	+ 深度学习Transformer自注意力机制: Self-attention$$ \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V $$
- **Python is extremely important.**
	the primary language for AI research & engineering, supporting framework implementation, data processing, model training, and deployment
- **Mathematics is fundamental.**
	- 线性代数 Linear Algebra
	- 微积分 Calculus
	- 概率论 Probability
- **Domain knowledge for your targeted AI field matters a lot.**

## 预备知识preliminaries

**从基本函数到深度学习**
![[Pasted image 20260612142106.png]]
1. 函数：机器学习 / 深度学习的数学本质，模型就是映射函数 `y = f(x)`，损失也是函数
2. 符号主义：早期 AI 流派，靠确定的逻辑、规则、推理（专家系统），不是深度学习。如：牛顿第二定律、勾股定理
3. 联结主义：神经网络核心思想，不再找到紧缺的函数关系，通过简化函数得到近似解。模仿人脑神经元连接，深度学习归属这一流派
4. 线性模型：最简单的拟合模型（线性回归、感知机），只能解线性可分问题，求近似解
5. 非线性模型：引入激活函数后的神经网络，解决线性不可分，是深度学习起点

### Linear model 线性模型

$$ f(x) = wx  + b $$
![[Pasted image 20260612152508.png]]
我们的目的就是使误差最小，因此引入损失函数来判断拟合度，
y和$\hat{y}_i$表示真实值与预测值的误差

$$ \sum_{i=1}^{N} |y_i - \hat{y}_i| $$
使用均方误差函数（MSE）优化损失函数
$$ L(w,b)=\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2 $$

+ 平方解决了绝对值函数不平滑的问题，也放大了误差较大得值得影响
+ $\frac{1}{n}$ 消除了样本数量大小的影响

MSE目标：求解让L最小的w和b的值
等价于：
使得每个参数的==偏导数==等于0，这就是深度学习中最基本的一种分析方法，**线性回归**
$$ \frac{\partial L(w,b)}{\partial w} = 0 \quad \frac{\partial L(w,b)}{\partial b} = 0 $$

### 非线性模型
![[Pasted image 20260612153327.png]]
由于线性模型的局限性，我们引入了非线性激活函数，g(x)被称作激活函数，用来解决线性函数拟合不了的非线性数据，可嵌套

一个最简单的神经网络：
$$ f(x) = g (wx  + b) $$
多层嵌套的神经网络
$$ f(x_1,x_2)=g(w_1x_1 + w_2x_2 + b) + b_2 $$
对于更复杂的神经网络，抽象为矩阵运算可以充分利用GPU的并行计算特性
![[Pasted image 20260612145527.png|300x200]]

标量形式： $$ \begin{align*} y_1 &= w_{11}x_1 + w_{12}x_2 + w_{13}x_3 + b_1 \\ y_2 &= w_{21}x_1 + w_{22}x_2 + w_{23}x_3 + b_2 \end{align*} $$
等价于
$$ \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} w_{11} & w_{12} & w_{13} \\ w_{21} & w_{22} & w_{23} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} + \begin{bmatrix} b_1 \\ b_2 \end{bmatrix} $$
 展开计算： $$ \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} w_{11}x_1 + w_{12}x_2 + w_{13}x_3 \\ w_{21}x_1 + w_{22}x_2 + w_{23}x_3 \end{bmatrix} + \begin{bmatrix} b_1 \\ b_2 \end{bmatrix} $$最终： $$ \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} w_{11}x_1 + w_{12}x_2 + w_{13}x_3 + b_1 \\ w_{21}x_1 + w_{22}x_2 + w_{23}x_3 + b_2 \end{bmatrix} $$矩阵表示
 $$ Y = g(WX+b)$$
抽象后得到神经网络前向传播的通用公式
 $$ A^{[L]} = g\left( W^{[L]} A^{[L-1]} + b^{[L]} \right) $$
- A[L]：第 L 层的激活值（输出）d
- W[L]：第 L 层的权重矩阵
- b[L]：第 L 层的偏置向量
- A[L−1]：上一层（第 L−1 层）的激活值（输入）
- g：激活函数（如 sigmoid、ReLU 等）


神经网络是一个线性函数和非线性激活函数不断组合形成的一个非常复杂的非线性函数，但神经网络并不是越复杂越好，在少数据量时容易
+ 过拟合
 + 泛化能力差
 > 泛化能力：对未知数据的预判能力

![[Pasted image 20260612143418.png]]

**解决过拟合问题**
从数据和模型本身入手：
对于小数据集，人为改动数据，如图片加滤镜、反转等，可以优化模型鲁棒性
>鲁棒性（Robostness）：使得模型能够在面对小变化时不产生大波动

从训练过程入手：
1. 提前终止任务
2. 正则化：像损失函数中添加权重惩罚项
	![[Pasted image 20260612144542.png]]
3. Dropout：训练过程中每次随机丢弃部分参数



### 监督学习
`原始文件/路径 → 数据索引&标签管理 → 统一为数值数组 → 模型专用张量 + 分批加载`

对应结构接力：
`list/tuple/dict(存路径 / 名称）→ DataFrame（标签 / 元数据）→ ndarray（数值化）→ Tensor + Dataset/DataLoader（模型输入）`
```python
#输入：特征 + 标签 
for imgs, labels in train_loader: 
	# 用标签计算损失，反向传播更新参数
	out = model(imgs)  
	loss = criterion(out, labels)
```

### 无监督学习

`list(路径/原始文本/分子串) → 数据清洗/筛选(可选，弱化DataFrame) → ndarray(数值化) → Tensor + DataLoader → 模型训练(挖掘数据内在结构)`
少了「标签绑定」这一步，其他数据结构接力逻辑照旧

```python
# 只有图像，没有标签 
for imgs in train_loader: 
	# 构造增广样本（造伪任务） 
	aug1, aug2 = augment(imgs), augment(imgs) 
	z1, z2 = model(aug1), model(aug2) 
	# 损失：让两个增广样本表征尽可能接近 
	loss = contrastive_loss(z1, z2)
```






## Convolutional Neural Network
全连接有很多问题
1. 时空复杂度很高
2. 无法理解图片局部问题
因此采用==Convolutional Neural Network(CNN)==来优化这个问题
![[Pasted image 20260612151635.png]]

CNN 是专门处理具有空间 / 时序结构数据（图像、语音、文本序列）的神经网络，核心是用「卷积」替代全连接，解决传统网络参数爆炸、无法捕捉局部特征的问题。
![[Pasted image 20260612152220.png]]
### 核心设计思想：为什么 CNN 适合图像？

1. **局部感受野（Local Receptive Field）**
    
    每个神经元只看输入的一小块区域（比如 3×3、5×5），不用一次性处理整张图，大幅减少计算量，也符合生物视觉的 “局部感知” 逻辑。
2. **权重共享（Weight Sharing）**
    
    同一个卷积核（滤波器）在整张图上滑动时，用同一套参数提取相同特征，不用为每个位置单独训练参数，参数量直接下降几个数量级。
3. **池化（Pooling）降维**
    
    对特征图做下采样，保留关键信息的同时缩小尺寸，减少计算量，还能提升模型对位置偏移的鲁棒性。

### CNN 核心模块拆解（按前向传播顺序）

1、卷积层（Convolutional Layer）—— 核心特征提取

- **核心：卷积核（滤波器）**
    
    通常是 3×3、5×5 或 1×1 的矩阵，里面的权重是可学习的。每个卷积核负责提取一种特定特征（比如边缘、纹理、颜色块）。
- **卷积运算过程**
    
    卷积核按「步长（Stride）」在输入特征图上滑动，每到一个位置，就和对应区域做加权求和，输出一个值；如果需要保持输入输出尺寸一致，会用「填充（Padding）」在图像边缘补 0。
- **多通道输出**
    
    一层卷积层通常有多个卷积核（比如 32、64、128 个），每个核输出一张特征图，合起来就是多通道的特征表示（比如 32 个核就输出 32 通道）。

2、激活层（Activation Layer）—— 引入非线性

卷积运算本质是线性操作，叠加激活函数（比如 ReLU、Leaky ReLU、Swish）才能让模型拟合复杂模式，否则多层卷积还是等价于一层线性变换。

- 常用 ReLU：`f(x) = max(0, x)`，简单高效，缓解梯度消失问题。

3、池化层（Pooling Layer）—— 降维 + 鲁棒性

- **最大池化（Max Pooling）**：取局部区域的最大值，最常用，保留最显著的特征。
- **平均池化（Average Pooling）**：取局部区域的平均值，对背景噪声更鲁棒。
- 典型设置：2×2 池化核，步长 2，特征图尺寸直接减半，计算量大幅下降。

4、全连接层（Fully Connected Layer）—— 特征到输出的映射

在网络的最后几层，把前面提取的特征图展平成一维向量，通过全连接层映射到输出空间（比如分类任务的类别数、回归任务的数值）。

- 现代 CNN（如 ResNet）常用「全局平均池化」替代全连接层，进一步减少参数量，避免过拟合。
## Transformer


### self-attention
让序列中每个词 / 像素，都能 “看到” 其他所有位置的信息，并学习它们的关联权重
$$ \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V $$
- Q（Query）：当前位置的 “查询向量”，表示 “我要关注谁”
- K（Key）：其他位置的 “键向量”，表示 “我是什么样的”
- V（Value）：其他位置的 “值向量”，表示 “我携带的信息”
- $d_k$：Key 向量维度，缩放因子，防止 softmax 后梯度消失

比如句子「我 吃 苹果」：

- 每个词都生成 \(Q/K/V\)
- 「吃」的 Q 会和「苹果」的 K 计算出高相似度，注意力权重变大，从而重点关注「苹果」的信息
- 最终输出的向量，就是根据权重加权后的所有词的信息融合

### Multi-Head Attention

单头注意力只能捕捉一种关联模式，多头注意力通过并行多个独立的注意力头，从不同维度建模不同类型的依赖

$$ \text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O $$

$$ \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V) $$



[^1]: 