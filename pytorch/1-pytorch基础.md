---
marp: false
---
# 数据类型

| type |    dtype    |    CPU tensor    |
| :---: | :---------: | :---------------: |
| bool | torch.bool | torch.BoolTensor |
| char | torch.int8 | torch.CharTensor |
| uchar | torch.uint8 | torch.ByteTensor |
| short | torch.int16 | torch.ShortTensor |

---

|  type  |     dtype     |     CPU tensor     |
| :----: | :-----------: | :----------------: |
|  int  |  torch.int32  |  torch.IntTensor  |
|  long  |  torch.int64  |  torch.LongTensor  |
| float | torch.float32 | torch.FloatTensor |
| double | torch.float64 | torch.DoubleTensor |

- **GPU tensor**即为 torch.cuda.`xxx`Tensor.
- 默认的整形为torch.LongTensor

---

# Tensor的性质

## Tensor和tensor的区别

**torch.Tensor**是类，它等于torch.FloatTensor，主要用做创建Tensor;
		**torch.tensor**是一个函数，会从data中的数据部分做拷贝（而不是直接引用），根据原始数据类型生成相应类型的数据。

---

## 基础创建

**ps：** '代表默认不取到

- `tenosr.Tensor([1,3])`    	从列表中获得相应值和形状
- `tenosr.Tensor(m,n,)`     	 返回shape为（m, n）的全0数据
- `torch.tensor()`
- `torch.DoubleTensor([1, 3])`      指定数据类型创建
- `torch.from_numpy(a)`         	        从numpy数据类型创建
  - 实际上直接torch.tensor()就可以

---

- `torch.zeros()`
- `torch.empty()`       		类似torch.zeros
- `torch.ones()`
- `torch.eye()`         		   创建单位阵
- `torch.full(size, n)`     						      创建tensor形状为size，以数n填充
- `torch.arange(start, stop', step)`    	step可选
- `torch.linspace(start, stop, n)`  	       等分取 n 个值，包含 stop
- `torch.logspace(start, stop, n, base=10)`   	等分，默认返回以10的n指数值

---

## 随机数

- `torch.rand()`  				从[0,1]均匀分布中随机抽取数据
- `torch.randn()` 	           标准正态分布的随机数，均值 0, 方差 1
- `torch.randint(low, high', size)`
- `torch.normal(mean, std， size)`  正态分布
- `torch.randperm(n',)` 	返回 0 到 n-1 之间，所有数字的一个随机排列
- `torch.rand_like(a)`       只模仿形状

---

## 数据类型

- `type(a)`         	->  返回大类 class
- `a.type`          	->  返回数据类型 CPU tensor
- `a.dtype`         	->  返回数据类型 dtype
- `isinstance(a, type)`    ->  返回bool值，判断数据类型

---

## 维度和形状

- `a.dim()`         		->  返回数据维度
- `a.shape[dim]`    	->  返回数据形状, 可指定dim
- `a.size(dim)`     	->  返回数据形状, 可指定dim
- `a.reshape() = a.view()`

---

## 数学运算

**ps:** 符合广播机制

- `torch.add(a, b)`     	->  $a + b$
- `torch.sub(a, b)`     	->  $a - b$
- `torch.mul(a, b)`  	     ->  $a * b$   点乘
- `torch.div(a, b)`     	->  $a / b$
- `torch.abs(a)`          	  ->  $|a|$
- `torch.sqrt(a)`         	 ->  $\sqrt[2]{a}$
- `torch.pow(a, n)`     	 ->  $a^{n}$
- `torch.exp(a)`          	   ->  $e^a$

---

数学运算：

- `torch.log(a)`         	->  $log_e(a)$
  可选 `torch.log2, torch.log10`
- `torch.ceil(a)`       	->  向上取整
- `torch.floor(a)`      	->  向下取整
- `torch.round(a)`      	->  四舍五入
- `torch.clamp(a，min，max)`    -> 数据裁剪至范围[min，max]

---

## 矩阵运算

- `torch.mm(a, b)`      ->  矩阵相乘（限制在二维）
- `torch.matmul(a, b) = a @ b`  ->  矩阵相乘
  矩阵乘法规则符合广播机制，且为最后两个低纬相乘
- ```python
  a = torch.rand(4, 3, 28, 64)
  b = torch.rand(4, 1, 64, 32)
  c = a @ b
  # c.shape = (4, 3, 28, 32)
  ```

---

## 属性统计

**注：** -dim指的是维度可选，默认是全部数据

- `a.sum(-dim)`         ->  总和
- `a.prod(-dim)`        ->  累乘
- `a.mean(-dim)`        ->  a的均值
- `a.numel()`           ->  返回数据的总个数

---

**a.min(-dim, -keepdim=False)**     ->  最小值
**a.max(-dim, -keepdim=False)**     ->  最大值
    在添加dim时会额外返回值对应维度的位置，即返回(value， index)，在keepdim=True时会锁住dim的维度信息

**a.topk(n, -dim=0, largest=False)**
    取对应维度dim的最大的n个数，返回其值和对应dim的位置,是a.max()和a.min()的进阶操作

**a.kthvalue(num, -dim=0, -keepdim=False)**
    取对应维度dim的第n小的数，返回其值和对应dim的位置

---

- ```python
  a = torch.Tensor([[2,3,4,1], [0,5,2,1]])
  a.max()   
  #tensor(5.)   FloatTensor
  
  x, y = a.max(dim=1)  
  #x = tensor([4., 5.]),    y = tensor([2, 1])
  
  x1, x2 = a.topk(2, dim=1)
  #x1 = tensor([[4., 3.],[5., 2.]])
  #x2 = tensor([[2, 1],[1, 2]])
  ```

---

**a.argmax(-dim)**
    最大值索引，默认打平成一维后返回
    在keepdim=True时会锁住dim的维度信息

**a.argmin(-dim)**
    最小值索引，默认打平成一维后返回

**a.norm(n, -dim)**
    a的范数，可指定维度，返回的shape中指定的维度会消失

---

- `torch.eq(a, b)`    	->  判断两个张量是否相等,相等位置返回True
- `torch.equal(a, b)`   	->  判断两个张量是否相等,相等时返回True
- `tensor.all(a)`     	->  如果张量中所有元素都是True, 才返回True
- `tensor.any(a)`       	->  如果张量中存在一个元素为True, 就返回True

**compare:** >, <, <=, >=, !=, ==
    对应位置返回 True 或False

---

## 高阶OP

**torch.where(condition, x, y)**
    condition：判断条件
    x：若满足条件，则取x中元素
    y：若不满足条件，则取y中元素

- 示例：
- ```python
  condition = torch.rand(3,2)
  x = torch.full((3,2), 2)
  y = torch.zeros(3,2)
  result = torch.where(condition>0.5, x, y)
  ```

---

- 结果：
- ```python
  condition:
    tensor([[0.3494, 0.6782],
           [0.4332, 0.1350],
           [0.3481, 0.3302]])
  
  result:
    tensor([[0., 2.],
            [0., 0.],
            [0., 0.]])
  ```

---

**torch.gather(input, dim, index)**
    从输入中的对应维度中根据索引index选取数据，gather在one-hot为输出的多分类问题中，可以把最大值坐标作为index传进去，然后提取到每一行的正确预测结果

- 示例：
- ```python
  b = torch.Tensor([[1,2,3],[4,5,6]])
  index_1 = torch.LongTensor([[0,1,1],[0,0,0]])
  index_2 = torch.LongTensor([[0,1],[2,0]])
  result1 = torch.gather(b, dim=0, index=index_1)
  result2 = torch.gather(b, dim=1, index=index_2)
  ```

---

- 结果：
- ```python
  b: tensor([[1., 2., 3.],
             [4., 5., 6.]])
  
  result1: tensor([[1., 5., 6.],
                   [1., 2., 3.]])
  
  result2: tensor([[1., 2.],
                   [6., 4.]])
  ```

---

## 索引和切片

**...** 代表包含所有，以代替 : , : ,

- 示例：
- ```python
  a = torch.arange(24).reshape(2,3,4)
  b = a[..., :2]
  
  b： tensor([[[ 0,  1],    #取了0,1两列
               [ 4,  5],
               [ 8,  9]],
  
               [[12, 13],
               [16, 17],
               [20, 21]]])
  ```

---

**torch.index_select(input, dim, index)**
    返回张量的索引值

- input: 需要索引的tensor数据
- dim: 指定维度索引
- index: 索引值，也是一个tensor

---

- 示例：
- ```python
  a = torch.arange(10).view(2,5)
  index=torch.tensor([1, 3])
  select=torch.index_select(a, 1, index)
  ```
- 结果：
- ```python
  a: tensor([[0, 1, 2, 3, 4],
             [5, 6, 7, 8, 9]])
  
  select: tensor([[1, 3],
                  [6, 8]])
  ```

---

**masked_select(input, mask)**
    返回张量的索引值,且变为1维

- input: 需要索引的tensor数据
- mask: 与input形状相同的torch.BoolTensor或torch.ByteTensor数据类型
- ```python
  a = torch.arange(10).reshape(2,5)
  #a中大于等于10的数变为True，小于为False
  mask = a.ge(6)   
  b = torch.masked_select(a,mask)
  
  #b： tensor([6,7,8,9])
  ```

---

**torch.take(input, index)**
  返回张量的索引值，按index的数值取值，且变成相应的形状.

示例：

- ```python
  a = torch.arange(10).reshape(2,5)
  b = torch.take(a, torch.tensor([[3,1],[9,6]]))
  
  b: tensor([[3,1],
             [9,6]])
  ```

---

**torch.squeeze(a, dim)， a.squeeze(dim)**
    将tensor中大小为1的维度删除，可指定dim (int类型)

**torch.unsqueeze(a, dim)， a.unsqueeze(dim)**
    扩充数据维度，可指定dim (int类型)

---

**a.expand(size)**
    将a在原shape数据上进行高维拓维或广播, size不能小于a.shape，不动的维度可以用-1代替

**a.repeat(size)**
    将a的shape按照size倍数重复扩张，会实际增加内存占用，不动的维度可以用-1代替

- ```python
  a = torch.rand(10).view(2,5)
  b = a.repeat(3,2).shape   #torch.Size([6, 10])
  ```

---

**torch.transpose(a, dim1, dim2), a.transpose(dim1, dim2)**
    维度交换，每次只能交换两个维度

**torch.permute(a, dim)， a.permute(dim)**
    维度变换，用dim排序所有维度

- ```python
  #bchw是torch图片存储格式，bhwc是numpy存储格式  
  a = torch.rand(4,3,28,32)     #图片格式bchw
  b = a.transpose(1,3)          #bwhc
  b = a.transpose(1,2)          #bhwc(4, 28, 32, 3)
  
  c = a.permute(a, (0, 2, 3, 1))    #bhwc
  ```

**ps:** `contiguous()`使内存连续

---

## 广播机制Broadcasting

numpy规则(适用pytorch):

1. 让所有输入数组都向其中形状最长的数组看齐，形状中不足的部分都通过在前面加 1 补齐
2. 如果输入数组的某个维度和输出数组的对应维度的长度相同或者其长度为 1时，这个数组能够用来计算，否则出错
3. 输出数组的形状是输入数组形状的各个维度上的最大值

---

简单理解：

- 将两个数组的维度太小右对齐，然后比较对应维度上的数值如果数值相等或其中有一个为1或者为空，则能进行广播运算输出的维度大小为取数值大的数值。否则不能进行数组运算

---

算数计算法:

- ```python
  a(4,1,3)   b(5,1)
  
  4 1 3
    5 1
  ----------
  4 5 3       最终数组运算大小为(4,5,3)
  若b为（5，2），则会因为不为1且与对面不相等而无法运算
  ```

---

## 合并与分割

**torch.cat(inputs, -dim=0)**
    合并输入的Tesor数据，只能在指定维度上不同，其他维度必须相同

- ```python
  a1 = torch.rand(4, 3, 32, 32)
  a2 = torch.rand(5, 3, 32, 32)
  a3 = torch.rand(1, 3, 32, 32)
  b = torch.cat((a1, a2, a3), dim=0)
  #b.shape(10, 3, 32, 32)
  ```

---

**torch.stack(inputs, -dim=0)**
    扩张再拼接输入的Tensor数据，会增加新的维度，可指定维度，数据shape必须完全相同

- ```python
  a1 = torch.rand(32, 8)
  a2 = torch.rand(32, 8)
  a3 = torch.rand(32, 8)
  b = torch.stack((a1, a2, a3), dim=1)
  #b.shape(32, 3, 8)
  ```

---

**torch.split(input, len/nums, -dim=0)**
**a.split(len/nums, -dim=0)**
    分割数据,可指定维度分割

- len： 按长度分割
- nums: 按数值分割

---

- 示例：
- ```python
  x = torch.randn(3,5)
  a1, a2 = a.split((2,1), dim=0)   #数值分割
  #a1.shape = (2, 5)
  #a2.shape = (1, 5)
  
  b1, b2, b3 = x.split(1, dim=0)    #长度分割
  #b1, b2, b3.shape = (1, 5)
  
  b = x.split(1, dim=0)     #则b为包含三个元素的元组
  ```

---

**torch.chunk(input, mem, -dim=0)**
**a.chunk(mem, -dim=0)**
    按数量进行分割

- 示例：
- ```python
  x = torch.randn(3,5)
  a1, a2 = a.split((2,1), dim=0)   #数值分割
  #a1.shape = (2, 5)
  #a2.shape = (1, 5)
  
  b1, b2, b3 = x.split(1, dim=0)    #长度分割
  #b1, b2, b3.shape = (1, 5)
  
  b = x.split(1, dim=0)     #则b为包含三个元素的元组
  ```

---
