# Enbedding嵌入层

## one hot编码

 假设，中文，一共只有10个字，**“我从哪里来，要到何处去”**

那么`one-hot`编码方式为: 列数为总体文字个数，行数为需要表示的文字数 

\# 我从哪里来，要到何处去

- [[1 0 0 0 0 0 0 0 0 0]
	[0 1 0 0 0 0 0 0 0 0]
	[0 0 1 0 0 0 0 0 0 0]
	[0 0 0 1 0 0 0 0 0 0]
	[0 0 0 0 1 0 0 0 0 0]
	[0 0 0 0 0 1 0 0 0 0]
	[0 0 0 0 0 0 1 0 0 0]
	[0 0 0 0 0 0 0 1 0 0]
	[0 0 0 0 0 0 0 0 1 0]
	[0 0 0 0 0 0 0 0 0 1]]

**即**：把每一个字都对应成一个十个（样本总数/字总数）元素的数组/列表，其中每一个字都用唯一对应的数组/列表对应，数组/列表的唯一性用1表示。如上，“我”表示成[1。。。。]，“去”表示成[。。。。1]，这样就把每一系列的文本整合成一个稀疏矩阵。

这样的编码，最大的好处就是，不管你是什么字，我们都能在一个一维的数组里用01给你表示出来。并且不同的字绝对不一样，以致于一点重复都没有，表达本征的能力极强。

**one-hot编码的优势**:  计算方便快捷、表达**本征**能力强。

**one-hot编码的缺点**:  表达**关联特征**的能力几乎为0，过于稀疏时，过度占用资源。	

- **比如**：中文大大小小简体繁体常用不常用有十几万，然后一篇文章**100W**字，则要表示成**100W X 10W**的矩阵

- 实际上，其实我们这篇文章，虽然 100W 字，但是其实我们整合起来，有99W字是重复的，只有1W字是完全不重复的。那我们用100W X 10W的岂不是白白浪费了99W X 10W的矩阵存储空间。

	<br>

## embedding

**1. 降维**

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9pbWcyMDE4LmNuYmxvZ3MuY29tL2Jsb2cvMTU0MDI0MC8yMDE5MDYvMTU0MDI0MC0yMDE5MDYyMjE0NDkwMDQ1Ny01MjExOTUzNTcuanBn?x-oss-process=image/format,png)

**embedding**通过矩阵乘法的方式，对稀疏矩阵进行降维，假如我们有一个 100W X10W 的矩阵，用它乘上一个10W X 20的矩阵，我们可以把它降到100W X 20，瞬间量级降10W/20=5000倍

<br>

**2. 升维**

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9pbWcyMDE4LmNuYmxvZ3MuY29tL2Jsb2cvMTU0MDI0MC8yMDE5MDYvMTU0MDI0MC0yMDE5MDYyMjE0NDcwNjIwOS05MDM3NzgyMTkuanBn?x-oss-process=image/format,png)

对于图片，距离的远近会影响我们的观察效果。同理低维的数据可能包含的特征是非常笼统的，我们需要不停地拉近拉远来改变我们的感受野，让我们对这幅图有不同的观察点，找出数据的特征。对低维的数据进行升维时，可能把一些其他特征给放大了，或者把笼统的特征给分开了。同时，这个embedding是一直在学习在优化的，就使得整个拉近拉远的过程慢慢形成一个良好的观察点。

为什么CNN层数越深准确率越高，卷积层卷了又卷，池化层池了又升，升了又降，全连接层连了又连。因为我们也不知道它什么时候突然就学到了某个有用特征。但是不管怎样，学习都是好事，所以让机器多卷一卷，多连一连，反正错了多少我会用交叉熵告诉你，怎么做才是对的我会用梯度下降算法告诉你，只要给你时间，你迟早会学懂。

因此，理论上，只要层数深，只要参数足够，NN能拟合任何特征。总之，它类似于虚拟出一个关系对当前数据进行映射。

<br>

**3. 关联性（密集态）**

对于one-hot编码one-hot编码，公主和王妃就变成了这样：

- 公 [0 0 0 0 1 0 0 0 0 0]
	主 [0 0 0 1 0 0 0 0 0 0]
	王 [0 0 0 0 0 0 0 0 0 1]
	妃 [0 0 0 0 0 0 0 0 1 0]

这样没有办法看出四行向量的内在联系。但是我们可以用其他特征表示：

|      | 皇帝 | 皇宫 |  女  |
| :--: | :--: | :--: | :--: |
| 公主 | 1.0  | 0.25 | 1.0  |
| 王妃 | 0.6  | 0.75 | 1.0  |

于是，我们就得出了公主和王妃的隐含特征关系：

- 王妃=公主的特征（1） * 0.6 +公主的特征（2） * 3 +公主的特征（3） * 1

我们把文字的one-hot编码，从稀疏态变成了密集态，并且让相互独立向量变成了有内在联系的关系向量。

<br>

**总结：**

- 它把我们的稀疏矩阵，通过一些线性变换（在CNN中用全连接层进行转换，也称为查表操作），变成了一个密集矩阵，这个密集矩阵用了N（例子中N=3）个特征来表征所有的文字，在这个密集矩阵中，表象上代表着密集矩阵跟单个字的一一对应关系，实际上还蕴含了大量的字与字之间，词与词之间甚至句子与句子之间的内在关系（如：我们得出的王妃跟公主的关系）。他们之间的关系，用的是嵌入层`embedding`学习来的参数进行表征。**从稀疏矩阵到密集矩阵的过程，叫做embedding**，很多人也把它叫做查表，因为他们之间也是一个一一映射的关系。

<br>

---

## `embedding`代码

```python
nn.Embedding(num_embeddings, embedding_dim, padding_idx=None)
```

**输入**（最重要的还是前三个参数）：

- `num_embeddings`,    – 词典的大小尺寸，比如总共出现5000个词，那就输入5000。此时index为（0-4999
- `embedding_dim`,      – 嵌入向量的维度，即用多少维来表示一个符号。
- `padding_idx=None`,– 填充id，比如，输入长度为100，但是每次的句子长度并不一样，后面就需要用统一的数字填充，而这里就是指定这个数字，这样，网络在遇到填充id时，就不会计算其与其它符号的相关性。（初始化为0）

**输出**:

- [规整后的句子长度，样本个数（batch_size）,词向量维度]
- `nn.Embedding`的shape，注意是会多添加一个`embedding_dim`维度：

**注**：

- 对句子进行规整，即对长度不满足条件的句子进行填充pad（填充的值也可以自己选定），另外句子结尾的EOS也算作一个词。
- 可以通过`weight`看对应的embedding字典矩阵对应的初始化数值，一般是通过正态分布进行初始化。

<br>

```python
# 嵌入字典的大小为10(即有10个词)，每个词向量的维度为3
embedding = nn.Embedding(10, 3)

# 该LongTensor的数字范围只能在0~9(因为设置了10)
input1 = torch.LongTensor([[1, 2, 4, 5], 
                           [4, 3, 2, 9]])
emb1 = embedding(input1)
```



```python
>>embedding.weight
>>tensor([[ 1.2402, -1.0914, -0.5382],		#每行是一个词汇的向量表示, 0
          [-1.1031, -1.2430, -0.2571],		# 1
          [ 1.6682, -0.8926,  1.4263],		# 2
          [ 0.8971,  1.4592,  0.6712],
          [-1.1625, -0.1598,  0.4034],
          [-0.2902, -0.0323, -2.2259],
          [ 0.8332, -0.2452, -1.1508],
          [ 0.3786,  1.7752, -0.0591],
          [-1.8527, -2.5141, -0.4990],
          [-0.6188,  0.5902, -0.0860]], requires_grad=True)

input1
>>tensor([[1, 2, 4, 5],
        [4, 3, 2, 9]])

emb1																			# emb1.shape = (2, 4, 3)
>>tensor([[[-1.1031, -1.2430, -0.2571],     
         	 [ 1.6682, -0.8926,  1.4263],    
         	 [-1.1625, -0.1598,  0.4034],
         	 [-0.2902, -0.0323, -2.2259]],
        	[[-1.1625, -0.1598,  0.4034],
         	 [ 0.8971,  1.4592,  0.6712],
         	 [ 1.6682, -0.8926,  1.4263],
         	 [-0.6188,  0.5902, -0.0860]]], grad_fn=<EmbeddingBackward>)
```

- `a=embedding(input1)`是去embedding.weight中取对应index的词向量！
	`emb1`的第一行，input处index=1，对应取出weight中index=1的那一行。**其实就是按index取词向量！**



<br><br>

---

# RNN循环神经网络

**优质形象RNN动画**：[循环神经网络](https://www.bilibili.com/video/BV1z5411f7Bm/?spm_id_from=333.337.search-card.all.click&vd_source=cb747e9ce5eec653e35e42849c7a811b)

**为什么需要RNN**： 比如时间序列数据，这类数据是在不同时间点上收集到的数据，反映了某一事物、现象等随时间的变化状态或程度。一般的神经网络，在训练数据足够、算法模型优越的情况下，给定特定的x，就能得到期望y。其一般处理单个的输入，前一个输入和后一个输入完全无关，但实际应用中，某些任务需要能够更好的处理序列的信息，即**前面的输入和后面的输入是有关系的**。 这时就要用到RNN网络



<img src="https://pic4.zhimg.com/v2-3884f344d71e92d70ec3c44d2795141f_r.jpg" alt="img" style="zoom:50%;" />

x是一个向量，它表示**输入层**的值

s/h是一个向量，它表示**隐藏层**的值, 它的形状从始至终都是不变的

U是输入层到隐藏层的**权重矩阵**	

o也是一个向量，它表示**输出层**的值

V是隐藏层到输出层的**权重矩阵**。

W是什么？**循环神经网络**的**隐藏层**的值s不仅仅取决于当前这次的输入x，还取决于上一次**隐藏层**的值s。**权重矩阵** W就是**隐藏层**上一次的值作为这一次的输入的权重。

<img src="https://pic2.zhimg.com/v2-b0175ebd3419f9a11a3d0d8b00e28675_r.jpg" alt="img" style="zoom: 33%;" />

这个网络在t时刻接收到输入$x_t$ 之后，隐藏层的值是$s_t$  ，输出值是$o_t$  。关键一点是， $s_t$ 的值不仅仅取决于$x_t$  ，还取决于 $x_t - 1$  。我们可以用下面的公式来表示**循环神经网络**的计算方法：

<img src="https://pic4.zhimg.com/v2-9524a28210c98ed130644eb3c3002087_r.jpg" alt="img" style="zoom: 33%;" />



---

## RNN 步骤

对于序列形的数据：

<img src="https://img-blog.csdnimg.cn/img_convert/de565ecf43e0bda278ce99eebd875322.png" alt="img" style="zoom:50%;" />



序列形的数据就不太好用原始的神经网络处理，为了建模序列问题，RNN引入了**隐状态h（即上文中的S）（hidden state）**的概念，**隐状态h可以对序列形的数据提取特征，接着再转换为输出**。

<br>

1. 先从$h_1$的计算开始看：

- 圆圈或方块表示的是向量。

- 一个箭头就表示对该向量做一次变换。如上图中$h_0$和$x_1$分别有一个箭头连接，就表示对$h_0$和$x_1$各做了一次变换。

<img src="https://img-blog.csdnimg.cn/img_convert/f22cb1de22144ad6806b83acb3fb45a4.png" alt="img" style="zoom: 80%;" />

<br>

2. 后续$h_2$的计算和$h_1$类似,但有以下注意事项:

- **在计算时，每一步使用的参数U、W、b都是一样的，也就是说每个步骤的参数都是共享的，这是RNN的重要特点**，

- 下文马上要看到的**LSTM中的权值则不共享**，因为它是在两个不同的向量中。而RNN的权值为何共享呢？很简单，因为RNN的权值是在同一个向量中，只是不同时刻而已。

<img src="https://img-blog.csdnimg.cn/img_convert/373128bfcb80a93d064f2eab01448f3c.png" alt="img" style="zoom: 67%;" />

<br>

3. 得到输出值的方法就是直接通过h进行计算：

<img src="https://img-blog.csdnimg.cn/img_convert/830d146d7e3e626b975a18ea59efc43a.png" alt="img" style="zoom: 67%;" />

<br>

4. 剩下的输出类似进行：

- 这就是最经典的RNN结构，是x1, x2, .....xn，输出为y1, y2, ...yn，也就是说，输入和输出序列必须要是等长的。

<img src=".\NLP\f6cdd1b5ff8c6ca0cad2f6afcea8f635.png" alt="img" style="zoom: 67%;" />



<br>

---

## RNN 多结构

RNN 有多种结构,如下所示：

<img src="https://imgconvert.csdnimg.cn/aHR0cHM6Ly9pbWFnZS5qaXFpemhpeGluLmNvbS91cGxvYWRzL2VkaXRvci8wNmFlZmNlZS02ZTc0LTRkZGUtYmVlMS01ZjgyYTViODVjOWUvMTU0NDc2MDc1ODIyNy5wbmc" alt="img" style="zoom: 67%;" />



<br>

---

## **RNN的局限**

**长期依赖问题 （Long-TermDependencies）**:

有时候，我们仅仅需要知道先前的信息来执行当前的任务。例如，我们有一个语言模型用来基于先前的词来预测下一个词。如果我们试着预测这句话中“the clouds are in the sky”最后的这个词“sky”，我们并不再需要其他的信息，因为很显然下一个词应该是sky。在这样的场景中，相关的信息和预测的词位置之间的间隔是非常小的，RNN可以学会使用先前的信息。

但是同样会有一些更加复杂的场景。比如我们试着去预测“I grew up in France...I speak fluent French”最后的词“French”。当前的信息建议下一个词可能是一种语言的名字，但是如果我们需要弄清楚是什么语言，我们是需要先前提到的离当前位置很远的“France”的上下文。这说明相关信息和当前预测位置之间的间隔就肯定变得相当的大。

不幸的是，在这个间隔不断增大时，RNN会丧失学习到连接如此远的信息的能力。

在理论上，RNN绝对可以处理这样的长期依赖问题。人们可以仔细挑选参数来解决这类问题中的最初级形式，但在实践中，RNN则没法太好的学习到这些知识。

**RNN 会受到短时记忆的影响。如果一条序列足够长，那它们将很难将信息从较早的时间步传送到后面的时间步。**

- **梯度会随着时间的推移不断下降减少，而当梯度值变得非常小时，就不会继续学习**。换言之， 在递归神经网络中，获得小梯度更新的层会停止学习—— 那些通常是较早的层。 由于这些层不学习，**RNN会忘记它在较长序列中以前看到的内容，因此RNN只具有短时记忆**。

RNN的变体——LSTM，可以在一定程度上解决梯度消失和梯度爆炸这两个问题！

<br>





---

## `RNN`代码

### nn.RNN()

```python
nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity=tanh, bias=True, batch_first=False, dropout=0, bidirectional=False)
```

**参数说明:**

- `input_size`  输入特征的维度， 一般rnn中输入的是词向量，那么 input_size 就等于一个词向量的维度
- `hidden_size`  隐藏层神经元个数，或者也叫输出的维度（因为rnn输出为各个时间步上的隐藏状态）,比如隐藏层维度是10，每个单词的embedding维度是20，通过隐藏层后把单词的embedding变成10维表示
- `num_layers`  网络的层数
- `nonlinearity `  激活函数，默认是tanh
- `bias`  是否使用偏置
- `batch_first`输入数据的形式，默认是 False，即形式**`(seq(num_step), batch, input_dim)`**，也就是将序列长度放在第一位，batch 放在第二位
- `dropout`是否应用dropout, 默认不使用，如若使用将其设置成一个0-1的数字即可
- `birdirectional`是否使用双向的 rnn，默认是 False

<br>

```python

# 输入特征维度100， 隐藏层维度20， 网络层个数1, 这里没加bias
rnn = nn.RNN(input_size=100, hidden_size=20, num_layers=1)			# rnn.shape = (100, 20)
# 输入batch_size=3, 10个序列， 序列的embedding=100
x = torch.randn(10, 3, 100)

# out_size, hiddle_size = rnn(input_size, hiddle_size)
# 这里的hidden_size(初始状态h0)可以不定义，因为根据input_size，和rnn结构可以推断
out, h = rnn(x, torch.zeros(1, 3, 20))

# 结果,(batch_size在中间)
# out.shape(seq_len, batch_size, input_size)
# h.shape(num_layers, batch_size, hidden_size)
out.shape = (10, 3, 20),  h.shape = (1, 3, 20)
```

$\because$ 根据公式：
$$
原公式：h_t = tanh(U\cdot X_t + W \cdot h_{t-1} + b ) 		\\
			H_t=tanh(input∗W_{xh}+H_{t−1}∗W_{hh}+bias)
$$
$\therefore$ 

- $W_{xh}$  **(U)**的维度为：`[hidden_size, input_size]`   —> $X @ W_{xh}^T$ (内部操作)

- $W_{hh}$  **(W)**的维度为：`[hidden_size, hidden_size]` — > $H_{t-1} @ W_{hh}^T$ (内部操作)

<br>

### nn.RNNCell()

```python
torch.nn.RNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype=None) 
```

上述的RNN是一整个流程，从第一个时间步到最后一个时间步都给我们集成好了。而一个Cell的RNN，也叫RNNCell，这个需要我们自己手动往后更新$h_i$计算每个时间步。

cell1的输入$x_t$是$X$在每个时间步上的数据，对时间步做一个for循环，$x_t$大小是：**`[batch_size,input_size]`**；同样，这只表示一个cell，则$h_1$的初始化也不用关心多少层是否双向的问题，其大小是：**`[batch_size,hidden_size]`**

cell1的输出也只有一个$h_1$，这个$h_1$用来更新隐藏层用来手动向后计算，大小**`[batch_size, hidden_size]`**



```python

x = torch.randn(10, 32, 100)			# (setp, batch_size, input_size)
cell = nn.RNNCell(100, 30)				# (input_size, hiddle_size)
h0 = torch.zeros(32, 30)					# (batch_size, hiddle_size)

out = []
for xt in x:
  hiddle = cell(xt, h0)
  out.append(hiddle)
  
# 结果
out.shape = (10)
output.shape = (32, 30)
```

<br>

## RNN + embedding综合

```python
import torch

# 使用RNN 有嵌入层和线性层
num_class = 4     # 4个类别
input_size = 4    # 输入维度是4=词典维度
hidden_size = 8   # 隐层是8个维度
embedding_size = 10 # 嵌入到10维空间
batch_size = 1
num_layers = 2    # 两层的RNN
seq_len = 5       # 序列长度是5

# 准备数据
idx2char = ['e','h','l','o'] # 字典
x_data = [[1,0,2,2,3]] # hello  维度（batch,seqlen）
y_data = [3,1,2,3,2] # ohlol    维度 (batch*seqlen)

# (batchsize,seqlen)
inputs = torch.LongTensor(x_data)
# label (batchsize*seqlen)
labels = torch.LongTensor(y_data)
# print(labels.shape)
# print(labels)

# 构造模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.emb = torch.nn.Embedding(input_size,embedding_size)
        self.rnn = torch.nn.RNN(input_size=embedding_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
        self.fc = torch.nn.Linear(hidden_size,num_class)

    def forward(self,x):
        hidden = torch.zeros(num_layers,x.size(0),hidden_size)
        x = self.emb(x) # (batch,seqlen,embeddingsize)
        x,_ = self.rnn(x,hidden)    # (1, 5, 8)
        x = self.fc(x)              # (1, 5, 4)
        return x.view(-1,num_class) # (5, 4)

model = Model()

# 损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)  # lr = 0.01学习太慢

# 训练
for epoch in range(15):
    optimizer.zero_grad()
    outputs = model(inputs) # inputs是(batch, seqlen) outputs是(batch*seqlen, num_class)
    loss = criterion(outputs,labels) # labels(5,)
    loss.backward()
    optimizer.step()

    _,idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    print("Predicted:",''.join([idx2char[x] for x in idx]),end='')
    print(",Epoch {}/15 loss={:.3f}".format(epoch+1,loss.item()))
```

<br><br>

---

# LSTM网络

`long and short-term memory`:



## 总体框架

<img src=".\NLP\watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FpYW45OQ==,size_16,color_FFFFFF,t_70.png" alt="img" style="zoom: 33%;" />

<img src=".\NLP\20190317220547737.png" alt="img" style="zoom: 67%;" />

<br>

## 结构分析

<img src=".\NLP\watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FpYW45OQ==,size_16,color_FFFFFF,t_70-1678876133179-10.png" alt="img" style="zoom: 50%;" />



**外部**：

- 输入： 细胞状态$C_{t-1}$，隐藏状态$h_{t-1}$，t 时刻输入向量$X_t$
- 输出： 细胞状态$C_t$，隐层状态$h_t$，
	- 其中$h_{t}$还作为 t 时刻的输出

**内部**：

1. 细胞状态$C_{t-1}$的信息， 一直在上面那条线传递， t时刻的隐层状态$h_t$与输入$X_t$会对$C_t$进行适当修改，然后传到下一层
2. $C_{t-1}$会参与t时刻输出$h_t$的计算
3. 隐藏状态$h_{t-1}$的信息， 通过**LSTM**的 “门” 结构，对细胞状态进行修改，并且参与输出的计算



## LSTM的输入输出

LSTM也是RNN的一种，输入基本没什么差别。通常我们需要一个时序的结构喂给LSTM，数据会被分成**t**个部分，也就是上面图里面的$X_t$, $X_t$可以看作是一个向量 ，在实际训练的时候，我们会用batch来训练，所以通常它的shape是**(batch_size, input_dim)**。

另外$C_0$与$h_0$的值，也就是两个隐层的初始值，一般是用全0初始化。





## LSTM的门结构

LSTM的门结构，是被设计出来的一些计算步骤，通过这些计算，来调整输入与两个隐层的值。

<img src="D:\Programming code\markdown\notebook\语音基础\NLP\20190317220701125.png" alt="在这里插入图片描述" style="zoom:50%;" />

![在这里插入图片描述](.\NLP\20190317220646219.png)

- 黄色图案代表神经元， 也就是$w^Tx+b$的操作， 区别在于激活函数不同，<font size=5>$\sigma$</font>代表`sigmoid`函数， $tanh$则是双曲线正切函数
- 粉色图案代表元素操作，+就是对应元素相加，X是对应元素相乘



### 遗忘门

<img src=".\NLP\watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FpYW45OQ==,size_16,color_FFFFFF,t_70-1678878195953-17.png" alt="在这里插入图片描述" style="zoom: 80%;" />

*<font size=5>$\sigma$</font>*的输出在0到1之间，这个输出$f_t$逐位与$C_{t-1}$的元素相乘，我们可以发现，当$f_t$的某一位的值为0的时候，$C_{t-1}$对应那一位的信息就被干掉了，而值为(0, 1)，对应位的信息就保留了一部分，只有值为1的时候，对应的信息才会完整的保留。因此，这个操作被称之为遗忘门。

<br>

### 更新门层

![在这里插入图片描述](.\NLP\watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FpYW45OQ==,size_16,color_FFFFFF,t_70-1678878575260-23.png)



$C_t$：可以看作是新的输入带来的信息，$tanh$这个激活函数将内容归一化到-1到1。

<font size=5>$i_t$</font>：看起来和遗忘门的结构是一样的，这里可以看作是新的信息保留哪些部分。

<br>

![在这里插入图片描述](.\NLP\watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FpYW45OQ==,size_16,color_FFFFFF,t_70-1678878598482-26.png)

遗忘门给出的$f_t*C_{t-1}$，表示过去的信息有选择的遗忘（保留）。右边也是同理.

新的信息$\tilde{C_t}*i_t$表示新的信息有选择的遗忘（保留），

最后再把这两部分信息加起来，就是新的状态$C_t$了。

<br>

### 输出门层

![在这里插入图片描述](.\NLP\watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FpYW45OQ==,size_16,color_FFFFFF,t_70-1678879185001-29.png)

此时细胞状态$C_t$已经被更新了，这里的$o_t$还是用了一个sigmoid函数，表示输出哪些内容，而$C_t$通过$tanh$缩放后与$o_t$相乘，这就是这一个timestep的输出了。

<br>

## 公式总结
$$
f_t = \sigma(W_f\cdot [h_{t-1},x_t] + b_f)\tag {遗忘门}
$$

$$
i_t = \sigma(W_i\cdot [h_{t-1},x_t] + b_i)\tag {更新门1,2} \\
\tilde{C_t} = tanh(W_c\cdot [h_{t-1},x_t] + b_c)
$$
<br>

$$
C_t = f_t * C_{t-1} + i_t * \tilde C_t \tag {Ct输出}
$$

$$
o_t = \sigma(W_0\cdot [h_{t-1},x_t] + b_0)	\tag {Ot输出}
$$

<br>
$$
h_t = o_t * tanh(C_t)	\tag {输出门层}
$$
<br>

## 参数计算

公式中：$W$有四个: $W_f,W_i,W_c,W_o$,  	$b$也是四个：$b_f,b_i,b_c,b_o$.......

<br>

---

## `nn.LSTM`代码

```python
lstm = nn.LSTM(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout, bidirectional)
```

**参数**：

- `input_size`：输入数据的特征维数，通常就是embedding_dim(词向量的维度)
- `hidden_size`：隐藏层的大小（即隐藏层节点数量），输出向量的维度等于隐藏节点数；
- `num_layers`：lstm 隐层的层数；
- `bias`：隐层状态是否带 bias，默认为 true；
- `batch_first`：默认值为：False（seq_len, batch, input_size），如果是 True，则 input 为(batch, seq, input_size)，
- `dropout`：默认值0，非零时，除最后一层，每一层的输出都进行dropout；
- `bidirectional`：默认为 False。如果设置为 True, 则表示双向 LSTM，

<br>

## nn.LSTM的输入输出格式

`nn.LSTM`中输入与输出关系为**`output, (hn, cn) = lstm(input, (h0, c0))`**，输入输出格式如下：

```python
输入数据格式：
input(seq_len, batch, input_size)
h0(num_layers * num_directions, batch, hidden_size)
c0(num_layers * num_directions, batch, hidden_size)
 
输出数据格式：
output(seq_len, batch, hidden_size * num_directions)
hn(num_layers * num_directions, batch, hidden_size)
cn(num_layers * num_directions, batch, hidden_size)
```

- $h_n$包含的是句子的最后一个单词（也就是最后一个时间步）的隐藏状态，$c_n$包含的是句子的最后一个单词的细胞状态，所以**它们都与句子的长度seq_len无关**。

- **output[-1]与h_n是相等的，因为output[-1]包含的正是batch_size个句子中每一个句子的最后一个单词的隐藏状态，注意LSTM中的隐藏状态其实就是输出，cell state细胞状态才是LSTM中一直隐藏的，记录着信息**

<br>

---

## LSTM变体GRU

它将忘记门和输入门合成了一个单一的更新门。同样还混合了细胞状态和隐藏状态，和其他一些改动。最终的模型比标准的LSTM模型要简单，也是非常流行的变体。GRU 的张量运算较少，因此它比 LSTM 的训练更快一些。

![](.\NLP\d90834b68537f786b8d8c9fe656cf37c.png)



pytorch中的对应公式：
$$
r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
    z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
    n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
    h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
$$

<br>

## `nn.GRU`代码

与nn.RNN相似

<br>

---

# encoder-decoder

编码-解码框架，目前大部分attention模型都是依附于Encoder-Decoder框架进行实现，在NLP中Encoder-Decoder框架主要被用来处理序列-序列问题。也就是输入一个序列，生成一个序列的问题。这两个序列可以分别是任意长度。



![图1](https://img-blog.csdnimg.cn/20200321173021798.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RpbmsxOTk1,size_16,color_FFFFFF,t_70)



**`encoder`:** 将现实问题转化为数学问题

- 以上图为例，输入$<x1,x2,x3,x4>$，通过RNN生成隐藏层的状态值$<h1,h2,h3,h4>$，如何确定语义编码C呢？最简单的办法直接用最后时刻输出的ht作为C的状态值，这里也就是可以用h4直接作为语义编码C的值，也可以将所有时刻的隐藏层的值进行汇总，然后生成语义编码C的值，这里就是$C=q(h1,h2,h3,h4)$，q是非线性激活函数。

**`decoder`:**  求解数学问题，并转化为现实世界的解决方案

- 解码器，根据输入的语义编码C，然后将其解码成序列数据，解码方式也可以采用RNN/LSTM/GRU/BiRNN/BiLSTM/BiGRU。Decoder和Encoder的编码解码方式可以任意组合，并不是说我Encoder使用了RNN，Decoder就一定也需要使用RNN才能解码，Decoder可以使用LSTM，BiRNN这些。

<br>

## 解码机制

[[论文1\]Cho et al., 2014 . Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.](https://arxiv.org/abs/1406.1078)

<img src=".\NLP\image-20230315201306104.png" alt="image-20230315201306104" style="zoom:80%;" />

论文1中指出，因为语义编码**C**包含了整个输入序列的信息，所以在解码的每一步都引入C。文中Ecoder-Decoder均是使用RNN，在计算每一时刻的输出yt时，都应该输入语义编码C，即$h_t = f(h_{t-1},y_{t-1},C),p(y_t) = f(h_t,y_{t-1},C)$。$h_t$为当前t时刻的隐藏层的值，$y_{t-1}$为上一时刻的预测输出，作为t时刻的输入，每一时刻的语义编码C都相同

<br>

[[论文2]Sutskever et al., 2014. Sequence to Sequence Learning with Neural Networks.](https://arxiv.org/abs/1409.3215)

<img src=".\NLP\image-20230315201956681.png" alt="image-20230315201956681" style="zoom:80%;" />

论文2方式相对简单，只在Decoder的初始输入引入语义编码C，将C作为隐藏层状态值$h_0$的初始值，$p(y_t) = f(h_t,y_{t-1})$。

<br>

**实际上两种方式都不合适:**

1. 如果按照论文1解码，每一时刻的输出如上式所示，从这里可以看出，在生成目标句子的单词时，不论生成哪个单词，是y1,y2也好，还是y3也好，他们使用的语义编码C都是一样的，没有任何区别。而语义编码C是由输入序列X的每个单词经过Encoder 编码产生的，这意味着不论是生成哪个单词，y1,y2还是y3，其实输入序列X中任意单词对生成某个目标单词yi来说影响力都是相同的，没有任何区别（如果Encoder是RNN的话，理论上越是后输入的单词影响越大，并非等权的，估计这也是为何Google提出Sequence to Sequence模型时发现把输入句子逆序输入做翻译效果会更好的原因）

2. 将整个序列的信息压缩在了一个语义编码C中，用一个语义编码C来记录整个序列的信息，序列较短还行，如果序列是长序列，比如是一篇上万字的文章，我们要生成摘要，那么只是用一个语义编码C来表示整个序列的信息肯定会损失很多信息，而且序列一长，就可能出现梯度消失问题，这样将所有信息压缩在一个C里面显然就不合理。

**既然一个C不行，那咱们就用多个C**

<br><br>

# Attention

## attention 原理

![](.\NLP\watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RpbmsxOTk1,size_16,color_FFFFFF,t_70-1678883558239-40.png)

上图引入了Attention 机制的`Encoder-Decoder`框架。不再只有一个单一的语义编码C，而是有多个C1,C2,C3这样的编码。当预测Y1时，可能Y1的注意力是放在C1上，那就用C1作为语义编码，当预测Y2时，Y2的注意力集中在C2上，那就用C2作为语义编码，以此类推，就模拟了人类的注意力机制。

<br>

##  语义编码$C_i$的计算

以机器翻译例子"汤姆追逐杰瑞" - **"Tom Chase Jerry"** 来说明：

- 当我们在翻译"杰瑞"的时候，为了体现出输入序列中英文单词对于翻译当前中文单词不同的影响程度，比如给出类似下面一个概率分布值：（Tom,0.3）（Chase,0.2）（Jerry,0.5）
- 每个英文单词的概率代表了翻译当前单词“杰瑞”时，注意力分配模型分配给不同英文单词的注意力大小。这对于正确翻译目标语单词肯定是有帮助的，因为引入了新的信息。同理，目标句子中的每个单词都应该学会其对应的源语句子中单词的注意力分配概率信息。这意味着在生成每个单词$Y_i$的时候，原先都是相同的中间语义表示C会替换成根据当前生成单词而不断变化的$C_i$。理解AM模型的关键就是这里，即由固定的中间语义表示C换成了根据当前输出单词来调整成加入注意力模型的变化的$C_i$。

而每个$C_i$可能对应着不同源句子单词的注意力分配概率分布，对于以上英语翻译来说，对应信息可能如下：
$$
C_{汤姆} = g(0.6*h(Tom),0.2*h(Chase),0.2*h(Jerry))	\\
C_{追逐} = g(0.2*h(Tom),0.7*h(Chase),0.1*h(Jerry))	\\
C_{杰瑞} = g(0.3*h(Tom),0.2*h(Chase),0.5*h(Jerry))	\\
$$
$h$就是对应隐藏层的值，g函数就是加权求和,  $a_{ij}$表示权值分布，那么$C_i$就可以表示为：
$$
C_i = \sum_{j=1}^{n} a_{ij}h_j
$$
<br>

## 分布概率$a_{ij}$的计算



<img src=".\NLP\watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RpbmsxOTk1,size_16,color_FFFFFF,t_70-1678935630253-43.png" alt="分布概率计算" style="zoom: 80%;" />

decoder上一时刻的输出值$Y_{i-1}$与上一时刻传入的隐藏层的值$S_{i-1}$通过RNN生成$H_i$，然后**计算$H_i$与$h1，h2，h3…hm$的相关性**，得到相关性评分$F_i = F(h_j, H_i) = [f1,f2,f3…fm]$，然后对$F_i$进行softmax就得到注意力分配$α_{ij}$。然后将encoder的输出值$h_i$与对应的概率分布$a_{ij}$进行点乘求和，得到注意力attention值$C_i = \sum_{j=1}^{n} a_{ij}h_j$

<br>

## Attention 机制的本质思想

**传统attention的$Q$来自外部**

一个典型的Attention思想包括三部分：**Q**: `query`、**K**: `key`、**V**: `value`。

- **$W^q$负责对“主角”进行线性变化，将其变换为$Q$，称为query**，
- **$W^k$负责对“配角”进行线性变化，将其变换为$K$，称为key**
- 通过计算Q与K之间的相关性a，得出不同的K对输出的重要程度；
- 再与对应的v进行相乘求和，就得到了Q的输出；

<img src=".\NLP\watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RpbmsxOTk1,size_16,color_FFFFFF,t_70-1678937286404-46.png" alt="Attention机制" style="zoom:80%;" />



参照上图可以这么来理解Attention，将`Source`中的构成元素想象成是由一系列的`<Key,Value>`数据对构成（对应到咱们上里面的例子，key和value是相等地都是encoder的输出值$h_i$），此时给定Target中的某个元素Query（对应到上面的例子也就是decoder中的$H_i$），通过计算Query和各个Key的相似性或者相关性，得到每个Key对应Value的权重系数，然后对Value进行加权求和，即得到了最终的Attention数值。所以本质上Attention机制是对Source中元素的Value值进行加权求和，而Query和Key用来计算对应**Value的权重系数**。
$$
Attention(Query,Source) = \sum _{i=1}^{L_x}Similiarity(Query,key_i) * value
$$
$L_x$表示source的长度, 对应上例$L_x = 3$

<br>



---

## 总结

<img src=".\NLP\5cbcff99ded34619b6770ca11e5e2b8c.png" alt="详细" style="zoom: 50%;" />



- step1，计算Q对每个K的相关性，即函数$F (Q, K)$；
	- 这里计算相关性的方式有很多种，常见方法比如有：
	- 求两者的【**向量点积**】$，Similarity(Q，Ki) = Q⋅Ki$。
	- 求两者的向量【**余弦相似度**】，$Similarity(Q，Ki)=Q⋅Ki∣∣Q∣∣⋅∣∣Ki∣∣$。
	- 引入一个**额外的神经网络**来求值，$Similarity(Q，Ki)=MLP(Q,Ki)$。
- step2，对step1的注意力的分进行归一化；
	- softmax的好处首先可以将原始计算分值整理成所有元素权重之和为1的概率分布；
	- 其次是可以通过softmax的内在机制更加突出重要元素的权重；
	- $a_i$即为$value_i$对应的权重系数;
- step3，根据权重系数对$Value$进行加权求和，即可求出针对$Query$的$Attention$数值。

$$
C_{输入} = Attention(Query,Source) = \sum _{i=1}^{L_x}Similiarity(Query,key_i) * value_i  \\
= \sum _{i=1}^{L_x}a_{i} * value_i  \\
$$

<br>
<br>

# Transformer

## transformer框架

[transformer解读1](https://blog.csdn.net/Tink1995/article/details/105080033?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522167889609516800184171782%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=167889609516800184171782&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-105080033-null-null.142^v73^pc_search_v2,201^v4^add_ask,239^v2^insert_chatgpt&utm_term=transformer&spm=1018.2226.3001.4187)

[transformer解读2](https://jalammar.github.io/illustrated-transformer/)

<img src=".\NLP\v2-0c259fb2d439b98de27d877dcd3d1fcb_r.jpg" style="zoom:80%;" />

<br>

## 案例

翻译： 机器学习 -> machine learning

<img src=".\NLP\06f9885a2606e5ee31c8fd0f6ee13e24.jpeg" style="zoom:80%;" />

<br>

- 黑盒子部分：

<img src="https://img-blog.csdnimg.cn/img_convert/cab7ef0bf968af5e1ff203df1f927829.jpeg" />

<br>

- **`encoders-decoders`**部分：

![img](.\NLP\1c983a9ac3629c13a2e7ac533af3d325.jpeg)

<br>

- **`encoder`**里的结构是一个**自注意力机制**（实际上是多头自注意力机制	）加上一个**前馈神经网络**。

![](.\NLP\cab7ef0bf968af5e1ff203df1f927829-1678954720104-83.jpeg)

`self-attention`的输出矩阵$Z$的维度是（序列长度×D词向量），之后前馈神经网络的输出也是同样的维度

<br>



## Self_Attention

**!必看**  [self-attention详解](https://blog.csdn.net/zhaohongfei_358/article/details/122861751 “atention详解”)

1. **与Attention的区别**：==self-Attention中的Q是对自身（self）输入的变换，而在传统的Attention中，Q来自于外部==

self-attention的输入就是词向量，即整个模型的最初的输入是词向量的形式。首先将词向量乘上三个矩阵，得到三个新的向量，之所以乘上三个矩阵参数而不是直接用原本的词向量是因为这样增加更多的参数，提高模型效果。<br>

对于翻译：机器学习 -> meanching learning

1. **求单词的$K,Q,V$**。对于输入X1(机器)，乘上三个矩阵后分别得到$Q1,K1,V1$，同样的，对于输入X2(学习)，也乘上三个不同的矩阵得到$Q2,K2,V2$。

<img src=".\NLP\99e844e4c95502a593a25df4d737757d.jpeg" alt="img" style="zoom: 67%;" />

<br>

2. **计算注意力得分**$F (Q, K)$，$Q$和$K$维度需要相同这里选择Q与各个单词的K向量的点积的方法，假设分别得到得分112和96。

<img src="https://img-blog.csdnimg.cn/img_convert/df211d5d165ec67573458d4ed1a44ee5.jpeg"  />

<br>

3. 稳定梯度(可选)

<img src=".\NLP\df19b57b4b572468f31d2257d5dbfbef.jpeg" style="zoom:80%;" />

<br>

4. **softamx归一化**, 得到单词权重$a_i$

5. **权重和$value_i$相乘，再相加**，得到该单词的$self-attention$输出Z(即$C_{机器}$),其余位置的self-attention输出以同方式计算。

	<img src="https://img-blog.csdnimg.cn/img_convert/6f254e40128b5630a2db9596f56f0ef6.jpeg" style="zoom:80%;" />

<br>

将上述的过程总结为一个公式就可以用下图表示：

<img src="https://pic4.zhimg.com/v2-752c1c91e1b4dbca1b64f59a7e026b9b_r.jpg" style="zoom:80%;" />



## Self_Attention 代码

**形状**：
$$
	I_{n \times d}		\\ \\
	
	Q_{n \times d_k} = I_{n \times d} \cdot W^{q}~_{n \times d_k}~ 	\\
	K_{n \times d_k} = I_{n \times d} \cdot W^{k}~_{n \times d_k}~ 	\\  \\
	V_{n \times d_v} = I_{n \times d} \cdot W^{v}~_{n \times d_v}~ 	\\
	
	Attention(Q,K,V)_{n \times d_v} = softmax(\frac {QK^T} {\sqrt {d_k}})V		\\
$$

- n:  序列长度， 

- d : 序列维度， embedding_num 

- dim_k一般等于d， $d_K$决定了线性层的宽度

- dim_v一般等于d，$d_v$为输出向量的维度

<br>

**`self_attention` 模型**：

```python
class SelfAttention(nn.Module):
    def __init__(self, input_vector_dim: int, dim_k=None, dim_v=None):
        """
        初始化SelfAttention，包含如下关键参数：
        input_vector_dim: 输入向量的维度，对应上述公式中的d，例如你将单词编码为了10维的向量，则该值为10
        dim_k: 矩阵W^k和W^q的维度
        dim_v: 输出向量的维度，即b的维度，例如，经过Attention后的输出向量b，如果你想让他的维度为15，则该值为15，若不填，则取input_vector_dim
        """
        super(SelfAttention, self).__init__()

        self.input_vector_dim = input_vector_dim
        # 如果 dim_k 和 dim_v 为 None，则取输入向量的维度
        if dim_k is None:
            dim_k = input_vector_dim
        if dim_v is None:
            dim_v = input_vector_dim

        """
        实际写代码时，常用线性层来表示需要训练的矩阵，方便反向传播和参数更新
        """
        self.W_q = nn.Linear(input_vector_dim, dim_k, bias=False)
        self.W_k = nn.Linear(input_vector_dim, dim_k, bias=False)
        self.W_v = nn.Linear(input_vector_dim, dim_v, bias=False)

        # 这个是根号下d_k
        self._norm_fact = 1 / np.sqrt(dim_k)

    def forward(self, x):
        """
        进行前向传播：
        x: 输入向量，size为(batch_size, input_num, input_vector_dim)
        """
        # 通过W_q, W_k, W_v矩阵计算出，Q,K,V
        # Q,K,V矩阵的size为 (batch_size, input_num, output_vector_dim)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # permute用于变换矩阵的size中对应元素的位置，
        # 即，将K的size由(batch_size, input_num, output_vector_dim)，变为(batch_size, output_vector_dim，input_num)
        # 0,1,2 代表各个元素的下标，即变换前，batch_size所在的位置是0，input_num所在的位置是1
        K_T = K.permute(0, 2, 1)

        # bmm是batch matrix-matrix product，即对一批矩阵进行矩阵相乘
        # bmm详情参见：https://pytorch.org/docs/stable/generated/torch.bmm.html
        atten = nn.Softmax(dim=-1)(torch.bmm(Q, K_T) * self._norm_fact）

        # 最后再乘以 V
        output = torch.bmm(atten, V)

        return output
                                   
# 定义50个为一批(batch_size=50)， 一次输入5个向量,输入向量维度为3，欲经过Attention层后，编码成5个4维的向量：
model = SelfAttention(3, 5, 4)
model(torch.Tensor(50,5,3)).size() 		# (50, 5, 4)                                  
```

<br>

---

## MultiHead Attention

**优点**：

1. 扩展了模型关注不同位置的能力，这对翻译一下句子特别有用，因为我们想知道“it”是指代的哪个单词。

2. 给了自注意力层多个“**表示子空间**”。对于多头自注意力机制，我们不止有一组$K,Q,V$权重矩阵，而是有**多组**（论文中使用8组），所以**每个编码器/解码器使用8个“头”**（可以理解为8个互不干扰自的注意力机制运算），每一组的$K,Q,V$都**不相同**。然后，得到8个不同的权重矩阵$Z$，每个权重矩阵被用来将输入向量投射到不同的表示子空间。<br>

经过多头注意力机制后，就会得到多个权重矩阵Z，我们将多个Z进行拼接就得到了self-attention层的输出：

<img src=".\NLP\5e54bdbf0a672f669044cfeac0e0f954.png" style="zoom:80%;" />



self-attention的输出即是前馈神经网络层的输入，然后前馈神经网络的输入只需要一个矩阵就可以了，不需要八个矩阵，因此我们需要把这8个矩阵压缩成一个,把这些**矩阵拼接起来然后用一个额外的权重矩阵与之相乘**即可。

<img src=".\NLP\720a574b977b21df5e3ee7cae4ce4a54.jpeg"  />



Z矩阵作为前馈神经网络的输入，矩阵的维度是（序列长度×D词向量），之后前馈神经网络的输出也是同样的维度。

<br>

**基本的Multihead Attention单元**：对于encoder来说就是利用这些基本单元叠加，其中key, query, value均来自前一层encoder的输出，即encoder的每个位置都可以注意到之前一层encoder的所有位置。

<img src=".\NLP\v2-3cd76d3e0d8a20d87dfa586b56cc1ad3_r.jpg" style="zoom: 67%;" />

<br>

## MultiHead Attention代码

```python
def attention(query, key, value):
    """
    计算Attention的结果。
    这里其实传入的是Q,K,V，而Q,K,V的计算是放在模型中的，请参考后续的MultiHeadedAttention类。

    这里的Q,K,V有两种Shape，如果是Self-Attention，Shape为(batch, 词数, d_model)，
                           例如(1, 7, 128)，即batch_size为1，一句7个单词，每个单词128维

                           但如果是Multi-Head Attention，则Shape为(batch, head数, 词数，d_model/head数)，
                           例如(1, 8, 7, 16)，即Batch_size为1，8个head，一句7个单词，128/8=16。
                           这样其实也能看出来，所谓的MultiHead其实就是将128拆开了。

                           在Transformer中，由于使用的是MultiHead Attention，所以Q,K,V的Shape只会是第二种。
    """

    # 获取d_model的值。之所以这样可以获取，是因为query和输入的shape相同，
    # 若为Self-Attention，则最后一维都是词向量的维度，也就是d_model的值。
    # 若为MultiHead Attention，则最后一维是 d_model / h，h为head数
    d_k = query.size(-1)
    # 执行QK^T / √d_k
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 执行公式中的Softmax
    # 这里的p_attn是一个方阵
    # 若是Self Attention，则shape为(batch, 词数, 次数)，例如(1, 7, 7)
    # 若是MultiHead Attention，则shape为(batch, head数, 词数，词数)
    p_attn = scores.softmax(dim=-1)

    # 最后再乘以 V。
    # 对于Self Attention来说，结果Shape为(batch, 词数, d_model)，这也就是最终的结果了。
    # 但对于MultiHead Attention来说，结果Shape为(batch, head数, 词数，d_model/head数)
    # 而这不是最终结果，后续还要将head合并，变为(batch, 词数, d_model)。不过这是MultiHeadAttention
    # 该做的事情。
    return torch.matmul(p_attn, value)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model):
        """
        h: head的数量
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        # 定义W^q, W^k, W^v和W^o矩阵。
        # https://blog.csdn.net/zhaohongfei_358/article/details/122797190
        self.linears = [
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model),
        ]

    def forward(self, x):
        # 获取Batch Size
        nbatches = x.size(0)

        """
        1. 求出Q, K, V，这里是求MultiHead的Q,K,V，所以Shape为(batch, head数, 词数，d_model/head数)
            1.1 首先，通过定义的W^q,W^k,W^v求出SelfAttention的Q,K,V，此时Q,K,V的Shape为(batch, 词数, d_model)
                对应代码为 `linear(x)`
            1.2 分成多头，即将Shape由(batch, 词数, d_model)变为(batch, 词数, head数，d_model/head数)。
                对应代码为 `view(nbatches, -1, self.h, self.d_k)`
            1.3 最终交换“词数”和“head数”这两个维度，将head数放在前面，最终shape变为(batch, head数, 词数，d_model/head数)。
                对应代码为 `transpose(1, 2)`
        """
        query, key, value = [
            linear(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (x, x, x))
        ]

        """
        2. 求出Q,K,V后，通过attention函数计算出Attention结果，
           这里x的shape为(batch, head数, 词数，d_model/head数)
           self.attn的shape为(batch, head数, 词数，词数)
        """
        x = attention(
            query, key, value
        )

        """
        3. 将多个head再合并起来，即将x的shape由(batch, head数, 词数，d_model/head数)
           再变为 (batch, 词数，d_model)
           3.1 首先，交换“head数”和“词数”，这两个维度，结果为(batch, 词数, head数, d_model/head数)
               对应代码为：`x.transpose(1, 2).contiguous()`
           3.2 然后将“head数”和“d_model/head数”这两个维度合并，结果为(batch, 词数，d_model)
        """
        x = (
            x.transpose(1, 2)
                .contiguous()
                .view(nbatches, -1, self.h * self.d_k)
        )

        # 最终通过W^o矩阵再执行一次线性变换，得到最终结果。
        return self.linears[-1](x)

      
           
# 定义8个head，词向量维度为512
model = MultiHeadedAttention(8, 512)
# 传入一个batch_size为2， 7个单词，每个单词为512维度
x = torch.rand(2, 7, 512)
# 输出Attention后的结果
print(model(x).size())  		# torch.Size([2, 7, 512])
   
```

<br>

---

## decoder部分

对于decoder来讲，我们注意到有两个与encoder不同的地方，一个是第一级的Masked Multi-head，另一个是第二级的Multi-Head Attention不仅接受来自前一级的输出，还要接收encoder的输出。

<img src=".\NLP\v2-40cf3d31c1c0dca24872bd9fc1fc429f_r.jpg" style="zoom:67%;" />

第一级decoder的key, query, value均来自前一层decoder的输出，但加入了Mask操作，即我们只能attend到前面已经翻译过的输出的词语，因为翻译过程我们当前还并不知道下一个输出词语，这是我们之后才会推测到的。

而第二级decoder也被称作encoder-decoder attention layer，即它的query来自于之前一级的decoder层的输出，但其key和value来自于encoder的输出，这使得decoder的每一个位置都可以attend到输入序列的每一个位置。

**总结**：k和v的来源总是相同的，q在encoder及第一级decoder中与k,v来源相同，在encoder-decoder attention layer中与k,v来源不同。

`Add`代表了Residual Connection，是为了解决多层神经网络训练困难的问题，通过将前一层的信息无差的传递到下一层，可以有效的仅关注差异部分，如resnet

`Norm`则代表了Layer Normalization，通过对层的激活值的归一化，可以加速模型的训练过程



​	