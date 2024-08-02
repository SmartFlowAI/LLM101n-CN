# 微小梯度（Micrograd）：神经网络训练的核心

  ## 介绍

  在现代人工智能和机器学习的世界中，神经网络训练是至关重要的一部分。微小梯度（Micrograd）项目提供了一个简化版的自动求导引擎，用于实现神经网络训练的核心算法——反向传播（backpropagation）。通过这个项目，我们可以了解如何从头开始构建一个自动求导引擎，并使用它来训练一个简单的神经网络。这篇文章将带你一步步了解Eureka Labs 推出的首门课程LLM101n中微小梯度项目的每个部分，帮助你掌握神经网络训练的基本概念。

  ## 自动求导引擎

  自动求导引擎是实现反向传播算法的关键。反向传播算法允许我们计算每个参数对最终输出的影响，从而指导我们如何更新这些参数以提高模型的性能。以下是实现自动求导引擎的代码：

  ```Python
  class Value:
      """ stores a single scalar value and its gradient """
  
      def __init__(self, data, _children=(), _op=''):
          self.data = data
          self.grad = 0
          # internal variables used for autograd graph construction
          self._backward = lambda: None
          self._prev = set(_children)
          self._op = _op # the op that produced this node, for graphviz / debugging / etc
  
      def __add__(self, other):
          other = other if isinstance(other, Value) else Value(other)
          out = Value(self.data + other.data, (self, other), '+')
  
          def _backward():
              self.grad += out.grad
              other.grad += out.grad
          out._backward = _backward
  
          return out
  
      def __mul__(self, other):
          other = other if isinstance(other, Value) else Value(other)
          out = Value(self.data * other.data, (self, other), '*')
  
          def _backward():
              self.grad += other.data * out.grad
              other.grad += self.data * out.grad
          out._backward = _backward
  
          return out
  
      def __pow__(self, other):
          assert isinstance(other, (int, float)), "only supporting int/float powers for now"
          out = Value(self.data**other, (self,), f'**{other}')
  
          def _backward():
              self.grad += (other * self.data**(other-1)) * out.grad
          out._backward = _backward
  
          return out
  
      def relu(self):
          out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
  
          def _backward():
              self.grad += (out.data > 0) * out.grad
          out._backward = _backward
  
          return out
  
      def tanh(self):
          out = Value(math.tanh(self.data), (self,), 'tanh')
  
          def _backward():
              self.grad += (1 - out.data**2) * out.grad
          out._backward = _backward
  
          return out
  
      def exp(self):
          out = Value(math.exp(self.data), (self,), 'exp')
  
          def _backward():
              self.grad += math.exp(self.data) * out.grad
          out._backward = _backward
  
          return out
  
      def log(self):
          # (this is the natural log)
          out = Value(math.log(self.data), (self,), 'log')
  
          def _backward():
              self.grad += (1/self.data) * out.grad
          out._backward = _backward
  
          return out
  
      def backward(self):
  
          # topological order all of the children in the graph
          topo = []
          visited = set()
          def build_topo(v):
              if v not in visited:
                  visited.add(v)
                  for child in v._prev:
                      build_topo(child)
                  topo.append(v)
          build_topo(self)
  
          # go one variable at a time and apply the chain rule to get its gradient
          self.grad = 1
          for v in reversed(topo):
              v._backward()
  
      def __neg__(self): # -self
          return self * -1
  
      def __radd__(self, other): # other + self
          return self + other
  
      def __sub__(self, other): # self - other
          return self + (-other)
  
      def __rsub__(self, other): # other - self
          return other + (-self)
  
      def __rmul__(self, other): # other * self
          return self * other
  
      def __truediv__(self, other): # self / other
          return self * other**-1
  
      def __rtruediv__(self, other): # other / self
          return other * self**-1
  
      def __repr__(self):
          return f"Value(data={self.data}, grad={self.grad})"
  ```

  这个 `Value` 类用于存储一个数值及其梯度，并定义了加法和乘法操作。每个操作都会记录其对输入值的影响，这样在反向传播时可以正确地更新梯度。

  ## 随机数生成器

  ```Python
  # class that mimics the random interface in Python, fully deterministic,
  # and in a way that we also control fully, and can also use in C, etc.
  class RNG:
      def __init__(self, seed):
          self.state = seed
  
      def random_u32(self):
          # xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
          # doing & 0xFFFFFFFFFFFFFFFF is the same as cast to uint64 in C
          # doing & 0xFFFFFFFF is the same as cast to uint32 in C
          self.state ^= (self.state >> 12) & 0xFFFFFFFFFFFFFFFF
          self.state ^= (self.state << 25) & 0xFFFFFFFFFFFFFFFF
          self.state ^= (self.state >> 27) & 0xFFFFFFFFFFFFFFFF
          return ((self.state * 0x2545F4914F6CDD1D) >> 32) & 0xFFFFFFFF
  
      def random(self):
          # random float32 in [0, 1)
          return (self.random_u32() >> 8) / 16777216.0
  
      def uniform(self, a=0.0, b=1.0):
          # random float32 in [a, b)
          return a + (b-a) * self.random()
  
  random = RNG(42)
  ```

  这个部分定义了一个随机数生成器，用于初始化神经网络的参数。简单来说，这就像我们从一个大箱子里随机抓出一个球，但每次抓球的方式都是确定的（伪随机）。

  ## 神经网络类

  ```Python
  class Module:
  
      def zero_grad(self):
          for p in self.parameters():
              p.grad = 0
  
      def parameters(self):
          return []
  
  class Neuron(Module):
  
      def __init__(self, nin, nonlin=True):
          r = random.uniform(-1, 1) * nin**-0.5
          self.w = [Value(r) for _ in range(nin)]
          self.b = Value(0)
          self.nonlin = nonlin
  
      def __call__(self, x):
          act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
          return act.tanh() if self.nonlin else act
  
      def parameters(self):
          return self.w + [self.b]
  
      def __repr__(self):
          return f"{'TanH' if self.nonlin else 'Linear'}Neuron({len(self.w)})"
  
  class Layer(Module):
  
      def __init__(self, nin, nout, **kwargs):
          self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
  
      def __call__(self, x):
          out = [n(x) for n in self.neurons]
          return out[0] if len(out) == 1 else out
  
      def parameters(self):
          return [p for n in self.neurons for p in n.parameters()]
  
      def __repr__(self):
          return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
  
  class MLP(Module):
  
      def __init__(self, nin, nouts):
          sz = [nin] + nouts
          self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]
  
      def __call__(self, x):
          for layer in self.layers:
              x = layer(x)
          return x
  
      def parameters(self):
          return [p for layer in self.layers for p in layer.parameters()]
  
      def __repr__(self):
          return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
  ```

  这个部分定义了几个类，用来构建神经网络：

  - `Module` 类：是所有神经网络模块的基类，提供了 `zero_grad` 和 `parameters` 方法。
  - `Neuron` 类：定义了一个神经元，它有权重和偏置，还可以选择使用非线性激活函数（如 `tanh`）。
  - `Layer` 类：定义了一层神经网络，它由多个神经元组成。
  - `MLP` 类：多层感知器，由多个层组成。

  ### 损失函数：负对数似然损失（NLL）

  ```Python
  def nll_loss(logits, target):
      # subtract the max for numerical stability (avoids overflow)
      max_val = max(val.data for val in logits)
      logits = [val - max_val for val in logits]
      # 1) evaluate elementwise e^x
      ex = [x.exp() for x in logits]
      # 2) compute the sum of the above
      denom = sum(ex)
      # 3) normalize by the sum to get probabilities
      probs = [x / denom for x in ex]
      # 4) log the probabilities at target
      logp = (probs[target]).log()
      # 5) the negative log likelihood loss (invert so we get a loss - lower is better)
      nll = -logp
      return nll
  ```

  这个函数计算预测值和实际标签之间的损失。简单来说，它告诉我们模型有多“错”。我们希望通过训练使这个值最小。

  ## 数据生成和训练

  ```Python
  def gen_data(n=100):
      pts = []
      for _ in range(n):
          x = random.uniform(-2.0, 2.0)
          y = random.uniform(-2.0, 2.0)
          # concentric circles
          # label = 0 if x**2 + y**2 < 1 else 1 if x**2 + y**2 < 2 else 2
          # very simple dataset
          label = 0 if x < 0 else 1 if y < 0 else 2
          pts.append(([x, y], label))
      # create train/val/test splits of the data (80%, 10%, 10%)
      tr = pts[:int(0.8*n)]
      val = pts[int(0.8*n):int(0.9*n)]
      te = pts[int(0.9*n):]
      return tr, val, te
  train_split, val_split, test_split = gen_data()
  
  # init the model: 2D inputs, 16 neurons, 3 outputs (logits)
  model = MLP(2, [16, 3])
  
  def eval_split(model, split):
      # evaluate the loss of a split
      loss = Value(0)
      for x, y in split:
          logits = model([Value(x[0]), Value(x[1])])
          loss += nll_loss(logits, y)
      loss = loss * (1.0/len(split)) # normalize the loss
      return loss.data
  
  # optimize using Adam
  learning_rate = 1e-1
  beta1 = 0.9
  beta2 = 0.95
  weight_decay = 1e-4
  for p in model.parameters():
      p.m = 0.0
      p.v = 0.0
  
  # train
  for step in range(100):
  
      # evaluate the validation split every few steps
      if step % 10 == 0:
          val_loss = eval_split(model, val_split)
          print(f"step {step}, val loss {val_loss}")
  
      # forward the network (get logits of all training datapoints)
      loss = Value(0)
      for x, y in train_split:
          logits = model([Value(x[0]), Value(x[1])])
          loss += nll_loss(logits, y)
      loss = loss * (1.0/len(train_split)) # normalize the loss
      # backward pass (deposit the gradients)
      loss.backward()
      # update with Adam
      for p in model.parameters():
          p.m = beta1 * p.m + (1 - beta1) * p.grad
          p.v = beta2 * p.v + (1 - beta2) * p.grad**2
          p.data -= learning_rate * p.m / (p.v**0.5 + 1e-8)
          p.data -= weight_decay * p.data # weight decay
      model.zero_grad()
  
      print(f"step {step}, train loss {loss.data}")
  ```

  这个部分生成一些随机数据点，分为训练集、验证集和测试集。然后定义一个多层感知器（MLP）模型，并通过训练过程对其进行训练。训练的过程包括以下步骤：

  1. **前向传播**：计算模型的输出。
  2. **计算损失**：计算模型输出与实际标签之间的损失。
  3. **反向传播**：计算损失相对于每个参数的梯度。
  4. **参数更新**：根据梯度更新参数，使损失减小。

  通过这个训练过程，模型会逐渐学会如何更好地预测数据点的标签。每次迭代中，我们都会进行前向传播计算输出，然后计算损失，接着进行反向传播计算梯度，最后更新模型参数。

  ## 总结

  微小梯度（Micrograd）项目展示了如何从头开始构建一个自动求导引擎，并使用它来训练一个简单的神经网络。通过这个项目，我们可以深入理解神经网络训练的核心概念和实现方式，为进一步学习和研究深度学习打下坚实基础。希望这篇文章能帮助你更好地理解神经网络训练的基本原理，并激发你对机器学习的兴趣。

# LLM101n-CN 共建共学计划

LLM101n-CN 共建共学计划是由机智流联合书生·浦语社区兴趣小组发起 LLM101n 中文版共建共学计划，旨在将顶级的 AI 学习资源带到中文社区。在公众号后台回复 “**101n**” 加入 LLM101n-CN 共建共学计划，也期待更多的友好社区合作伙伴加入此计划！也欢迎关注中文版 repo：

<https://github.com/SmartFlowAI/LLM101n-CN>

<p align="center">
  <img width="500" alt="" src="https://github.com/user-attachments/assets/9c9d164c-443d-4d13-9e10-798a7c3ac571">
</p>
