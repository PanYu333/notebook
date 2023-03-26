---
marp: true
---
---

# 梯度grad

PyTorch提供两种求梯度的方法：`backward()` and `torch.autograd.grad()` ，他们的区别在于前者是给叶子节点填充.grad字段，而后者是直接返回梯度。
1. 无论`backward`还是`autograd.grad`在计算一次梯度后图就被释放了，如果想要保留，需要添加retain_graph=True

---

## backward()

- ```python
  #对于y =a*b = (x+1)*(x+2)
  x = torch.tensor(2., requires_grad=True)

  a = torch.add(x, 1)
  b = torch.add(x, 2)
  y = torch.mul(a, b)

  y.backward()
  print(x.grad)
  >>>tensor(7.)
  ```

---

- ```python
  #查看tensor属性
  print("requires_grad: ", x.requires_grad, a.requires_grad, b.requires_grad, y.requires_grad)
  print("is_leaf: ", x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)
  print("grad: ", x.grad, a.grad, b.grad, y.grad)

  >>>requires_grad:  True True True True
  >>>is_leaf:  True False False False
  >>>grad:  tensor(7.) None None None
  ```

---

使用 `backward()`函数反向传播计算tensor的梯度时，并不计算所有tensor的梯度，而是只计算满足这几个条件的tensor的梯度：

1. 类型为叶子节点、
2. requires_grad=True
3. 依赖该tensor的所有tensor的requires_grad=True。
   所有满足条件的变量梯度会自动保存到对应的grad属性里。

![对于y =ab = (x+1)(x+2)](https://pic1.zhimg.com/v2-fac592361177c47b2094d5bab9b58c30_r.jpg)

---

## autograd.grad()

- ```python
  x = torch.tensor(2., requires_grad=True)

  a = torch.add(x, 1)
  b = torch.add(x, 2)
  y = torch.mul(a, b)

  grad = torch.autograd.grad(outputs=y, inputs=x)
  print(grad, grad[0])
  >>>(tensor(7.),), tensor(7.)
  ```

- 因为指定了输出y，输入x，所以返回值就是 $\partial x/\partial y $ 这一梯度，完整的返回值其实是一个元组，保留第一个元素就行。

---

# 激活activate

## softmax
p = F.sotfmax(x)