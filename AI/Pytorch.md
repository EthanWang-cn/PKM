ML & DL all numerical calculations and model operations are essentially operating N-dimensional arrays

- 主角：Tensor→ 干计算、跑模型
- 配角：Python 数据结构、语法 + Pytorch/Pandas/Numpy → 管数据组织、读取、预处理、配置

Tensor张量是一个数学的概念，任何维度都可以叫做张量，几何代数中定义的张量是基于向量和矩阵的推广，比如我们可以将标量视为零阶张量，矢量可以视为一阶张量，矩阵就是二阶张量

![[Pasted image 20260614091938.png]]
##### Installation
GPU： CUDA 12.6
```pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126```
CPU
```pip install torch torchvision```
## Data Manipulation  
```python
# 查看PyTorch版本
print("PyTorch 版本:", torch.__version__)
# 关键：检查CUDA是否可用
print("CUDA 是否可用:", torch.cuda.is_available()) 
# 如果CUDA可用，会输出设备信息
if torch.cuda.is_available():
    print("CUDA 版本:", torch.version.cuda)
    print("GPU 设备:", torch.cuda.get_device_name(0))
else:
    # CUDA不可用自动使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前只能使用 {device}")
```
### Pytorch实现张量相关操作
```python 
import torch 
#一、创建各类张量 ===================== 
# 空张量（未初始化） 
empty_tensor = torch.empty(2, 3) 
# 全0张量 
zero_tensor = torch.zeros(2, 3) 
# 全1张量 
one_tensor = torch.ones(2, 3) 
# 自定义填充值张量 
full_tensor = torch.full((2, 3), fill_value=8) 
# 从列表创建张量 
list_data = [1, 2, 3, 4] 
from_list = torch.tensor(list_data) 
# [0,1) 均匀分布随机浮点 
rand_tensor = torch.rand(3, 4) 
# 标准正态分布浮点(仅支持浮点类型) 
randn_tensor = torch.randn(3, 4) 
# 随机整数 
[low, high) randint_tensor = torch.randint(low=0, high=20, size=(3, 4)) 
# 等差数列 
arange_tensor = torch.arange(0, 10, step=2) 

#二、查看张量基本信息 ===================== 
test_tensor = torch.randn(2, 3, 4) print("形状:", test_tensor.shape) print("维度数:", test_tensor.ndim) print("总元素数:", test_tensor.numel()) print("数据类型:", test_tensor.dtype) print("是否在GPU:", test_tensor.is_cuda) 

#三、形状变换 ===================== 
a = torch.ones(2, 3, 4) 
# reshape 重塑形状，-1 自动计算维度 
b1 = a.reshape(-1, 4) 
b2 = a.reshape(3, -1) 
# view 用法同 
reshape b3 = a.view(-1) 
# 多维张量展平 
b4 = a.flatten() 
# unsqueeze 增加维度 
c = torch.rand(3) 
c1 = c.unsqueeze(0) 
c2 = c.unsqueeze(1) 
# squeeze 删除维度为1的维度 
d = torch.rand(1, 3, 1, 4) 
d1 = d.squeeze() 

#四、索引 & 切片 ===================== 
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) 
row0 = x[0] 
# 取单行 
elem = x[1, 2].item() 
# 取单个数值 
col1 = x[:, 1] 
# 所有行，第1列 
part = x[0:2, :] 
# 前两行，所有列 

#五、逐元素四则运算 ===================== 
t1 = torch.tensor([1, 2, 3]) 
t2 = torch.tensor([4, 5, 6]) 
add = t1 + t2 
sub = t1 - t2 
mul = t1 * t2 
div = t1 / t2 
scalar_add = t1 + 10 # 张量+标量 

#六、矩阵乘法 ===================== 
m1 = torch.tensor([[1, 2], [3, 4]]) 
m2 = torch.tensor([[5, 6], [7, 8]]) 
elem_mul = m1 * m2 
# 逐元素相乘 
mat_mul1 = m1 @ m2 
# 标准矩阵乘法 
mat_mul2 = torch.matmul(m1, m2) 

#七、张量拼接 ===================== 
a_cat = torch.rand(2, 3) 
b_cat = torch.rand(2, 3) 
cat_dim0 = torch.cat([a_cat, b_cat], dim=0) # 按行拼接 
cat_dim1 = torch.cat([a_cat, b_cat], dim=1) # 按列拼接 

#八、数据类型转换 在===================== 
float_t = torch.rand(2, 2) 
int_t = float_t.int() 
long_t = float_t.long() 
float32_t = float_t.float() 
int_tensor = torch.ones(2, 2, dtype=torch.int) 

#九、CPU/GPU设备切换 ===================== 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print("当前设备:", device) 
dev_tensor = torch.rand(3, 4).to(device) 
cpu_tensor = dev_tensor.cpu() 

#十、常用统计运算 ===================== 
stat_t = torch.tensor([[1, 3, 5], [2, 4, 6]]) 
print("最大值:", stat_t.max()) 
print("最小值:", stat_t.min()) 
print("求和:", stat_t.sum()) 
print("均值:", stat_t.float().mean()) 
print("标准差:", stat_t.float().std()) 

#十一、与Numpy等Python对象互转
A = X.numpy()
B = torch.tensor(A)
type(A), type(B)
#要将大小为1的张量转换为Python标量，我们可以调用`item`函数或Python的内置函数
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```
### 广播机制





### 内存问题
```python
#在下面的例子中，我们用Python的`id()`函数演示了这一点， 它给我们提供了内存中引用对象的确切地址。 运行`Y = Y + X`后，我们会发现`id(Y)`指向另一个位置。 这是因为Python首先计算`Y + X`，为结果分配新的内存，然后使`Y`指向内存中的这个新位置。
before = id(Y)
Y = Y + X
id(Y) == before
#这可能是不可取的，原因有两个：
#1. 首先，我们不想总是不必要地分配内存。在机器学习中，我们可能有数百兆的参数，并且在一秒内多次更新所有参数。通常情况下，我们希望原地执行这些更新；
#2. 如果我们不原地更新，其他引用仍然会指向旧的内存位置，这样我们的某些代码可能会无意中引用旧的参数。

#幸运的是，执行原地操作非常简单。 我们可以使用切片表示法将操作的结果分配给先前分配的数组，例如`Y[:] = <expression>`。 为了说明这一点，我们首先创建一个新的矩阵`Z`，其形状与另一个`Y`相同， 使用`zeros_like`来分配一个全0的块。
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
#如果在后续计算中没有重复使用`X`， 我们也可以使用`X[:] = X + Y`或`X += Y`来减少操作的内存开销。
before = id(X)
X += Y
id(X) == before
```
## Data Preprocessing
