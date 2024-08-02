作者：格陵兰岛的虎

引言：欢迎加入 OpenAI 联合创始人硬核推出的 LLM101n 课程的中文版共建共学计划！本期重磅推出针对 mlp 模块的 pytorch 版实现代码的深度解读

LLM101n 是 OpenAI 联合创始人、“计算机视觉教母”李飞飞教授的高徒Andrej Karpathy 推出的“世界上显然最好的 AI 课程”。我们邀请了「机智流」的社区同学，并制作了本期针对 LLM101n 中关于如何基于 pytorch 实现多层感知机（MLP）的深度解读。我们后续还会更新关于该课程核心代码的解读，欢迎关注。

中文版共建仓库：

https://github.com/SmartFlowAI/LLM101n-CN

完整代码：

https://github.com/EurekaLabsAI/mlp/blob/master/mlp_pytorch.py

LLM101n 原始仓库：

https://github.com/EurekaLabsAI

---

代码目录结构树：

```
mlp
|-- README.md
|-- common.py
|-- data
|   |-- preprocess.py
|   |-- test.txt
|   |-- train.txt
|   `-- val.txt
|-- mlp_numpy.py
`-- mlp_pytorch.py
```

今天将和大家一起学习 LLM101n 课程中 MLP 部分的 Python 核心代码（pytorch版），即上面👆结构树中的 `mlp_pytorch.py`。大家可以使用`git clone`命令克隆好仓库，结合源代码和本解读一起食用更佳哦~

阅读 Tips：本文代码块的几乎每一句都有简短的注释哦~

# 代码解读

## 代码整体框架

如代码整体框架图所示，代码可以拆解为以下几个主要模块：



![](https://files.mdnice.com/user/58235/9422ccd2-a724-4272-b7fd-a4d9f89dad11.PNG)


-   dataloader 模块：为模型的训练、验证和测试加载数据。
-   模型定义模块：定义 MLP 模型，包括了使用`nn.Module`和 不使用`nn.Module`两种模型版本。在该模块下，使用了自定义的 RNG 随机数模块来进行模型参数的初始化。
-   模型训练模块：使用训练集训练 NLP 模型。
-   评估模块：评估模型表现。
-   模型推理模块：进行前向推理。
-   RNG 随机数模块：是一个自定义的随机数模块，用于控制随机数的生成和模型参数初始化，保证实验的重复性。

下面，我们将从这些模块的基础上出发解读代码。

> 注：使用到的第三方库：math、time、torch

## dataloader

定义了一个名为 dataloader 的函数，其接受三个参数：

-   tokens：一个包含所有 token 的列表或数组。
-   context_length：上下文的长度，即每次输入的 token 数量。
-   batch_size：每个批次的大小。

其定义代码如下：

```python
def dataloader(tokens, context_length, batch_size):
    # returns inputs, targets as torch Tensors of shape (B, T), (B, )
    n = len(tokens) # 计算 tokens 的长度 n，用于后续的遍历。
    inputs, targets = [], [] # 创建空的列表 inputs 和 targets 用于存储输入数据和目标数据。
    pos = 0 # 定义 pos 变量，表示当前窗口的起始位置。
    while True: # 进入一个 while 循环，用于不断生成批次数据。
        # simple sliding window over the tokens, of size context_length + 1
        window = tokens[pos:pos + context_length + 1] # 取从当前 pos 开始的 context_length + 1 个 token 作为窗口。
        inputs.append(window[:-1]) # 取窗口中的前 context_length 个 token 作为输入，并将它们添加到 inputs 列表中。
        targets.append(window[-1]) # 取窗口中的最后一个 token 作为目标，并将它添加到 targets 列表中。
        # once we've collected a batch, emit it
        if len(inputs) == batch_size: # 当 inputs 列表的长度等于 batch_size 时，生成当前批次的输入和目标张量。
            yield (torch.tensor(inputs), torch.tensor(targets)) # 使用 yield 关键字返回它们。此时 dataloader 函数成为一个生成器，能够在训练过程中按需提供数据。
            inputs, targets = [], [] # 重置 inputs 和 targets 列表以收集下一个批次的数据。
        # advance the position and wrap around if we reach the end
        pos += 1 # 将 pos 前移一个 token。
        if pos + context_length >= n: # 如果 pos 加上 context_length 超出了 tokens 的长度，则将 pos 重置为 0，从头开始循环。
            pos = 0
```

`dataloader`函数通过滑动窗口的方法，生成一系列输入和目标对，并按批次大小生成数据。

> 后记：之所以这段代码中的 `while` 循环体不需要退出条件，是因为其中的 `yield`语句在函数中创建了一个生成器（generator），它可以暂停函数的执行并返回一个值，同时保留函数的状态。下次迭代时，函数会从上次暂停的地方继续执行，而不是重新开始。这样就可以实现按需生成数据批次，而不需要一次性生成所有批次的数据，节省了内存空间并提高了效率。

# 模型定义

作者 Andrej Karpathy 在该仓库中使用了两种方式来构建 MLP 模型，区别在于实现过程中是否使用`torch.nn.Module`模块，但两种方法的一般思路都是初始化模型及其参数（实现`__init__()`）并定义前向传播函数（实现`forward()`）计算输出。

## MLPRaw

我们先不使用`nn.Module`而是通过手动初始化模型参数（如权重和偏置）的方法来定义一个 MLPRaw 模型类。

**首先**，手动初始化模型及其参数（也就是把搭建模型需要的积木准备好），代码如下：

```python
class MLPRaw: 
    def __init__(self, vocab_size, context_length, embedding_size, hidden_size, rng):       
        # 接受五个参数：
        ## vocab_size（词汇表大小）
        ## context_length（上下文长度）
        ## embedding_size（嵌入层大小）
        ## hidden_size（隐藏层大小）
        ## rng（随机数生成器，这个会在后面详细介绍） 。
        v, t, e, h = vocab_size, context_length, embedding_size, hidden_size    # 通过变量 v、t、e 和 h 分别表示词汇表大小、上下文长度、嵌入层大小和隐藏层大小。        
        self.embedding_size = embedding_size # 保存 embedding_size 为类的属性。
        # self.wte 表示嵌入层权重。
        ## 先使用随机数生成器 rng 生成一个正态分布 N(0, 1) 的随机数，并将其转换为 PyTorch 张量。
        ## 再将张量调整为形状 (v, e)，表示词汇表中的每个词都有一个 embedding_size 维度的嵌入向量。
        self.wte = torch.tensor(rng.randn(v * e, mu=0, sigma=1.0)).view(v, e) 
        scale = 1 / math.sqrt(e * t) # 计算第一个全连接层的缩放系数 scale，值为 1 / sqrt(e * t)。
        # self.fc1_weights 表示第一个全连接层的权重，self.fc1_bias 表示第一个全连接层的偏置。
        ## 使用随机数生成器 rng 生成均匀分布 U(-scale, scale) 的 t * e * h 个随机数，并将其转换为 PyTorch 张量。
        ## 将张量调整为形状 (h, t * e) 并转置，表示第一个全连接层的权重。
        self.fc1_weights =  torch.tensor(rng.rand(t * e * h, -scale, scale)).view(h, t * e).T
        self.fc1_bias = torch.tensor(rng.rand(h, -scale, scale))
        scale = 1 / math.sqrt(h) # 计算第二个全连接层的缩放系数 scale，值为 1 / sqrt(h)。
        # self.fc2_weights 表示第二个全连接层的权重，self.fc2_bias 表示第二个全连接层的偏置。
        ## 参数初始化方式如 fc1
        self.fc2_weights = torch.tensor(rng.rand(v * h, -scale, scale)).view(v, h).T
        self.fc2_bias = torch.tensor(rng.rand(v, -scale, scale))
        for p in self.parameters(): # 遍历 parameters 方法返回的所有模型参数，将它们的 requires_grad 属性设置为 True，表示这些参数需要计算梯度。
            p.requires_grad = True
    def parameters(self):
        # 定义 parameters 方法，返回模型的所有参数。
        return [self.wte, self.fc1_weights, self.fc1_bias, self.fc2_weights, self.fc2_bias]
```

初始化网络参数使用到的随机数生成器 rng 定义在`common.py`文件中，该 RNG 类提供了一个完全确定性的随机数生成器，实现了均匀分布或正态分布随机数的生成。通过 Box-Muller 变换和`xorshift*`算法实现生成正态分布和均匀分布的随机数，确保了生成器的可重复性和可控性。

RNG 类代码如下：

```python
def box_muller_transform(u1, u2):
    # 实现了基本形式的 Box-Muller 变换，用于将两个 [0, 1) 区间上的均匀随机数 u1 和 u2，
    # 转换为标准正态分布的随机数 z1 和 z2。
    z1 = (-2 * log(u1)) ** 0.5 * cos(2 * pi * u2)
    z2 = (-2 * log(u1)) ** 0.5 * sin(2 * pi * u2)
    return z1, z2

class RNG:
    def __init__(self, seed):
        # 接受一个种子 seed，用于初始化随机数生成器的状态 self.state，使得生成器是确定性的。
        self.state = seed

    def random_u32(self):
        # 实现了 xorshift* 随机数生成算法。
        ## 使用按位操作（^ 和 >>, <<）更新 self.state。
        ## 返回一个 32 位无符号整数，确保输出在 [0, 2^32-1] 范围内。
        # 使用 & 0xFFFFFFFFFFFFFFFF 确保结果在 64 位范围内（类似于在 C 中将结果强制转换为 uint64）。
        self.state ^= (self.state >> 12) & 0xFFFFFFFFFFFFFFFF 
        # 上一步的操作：
        ## 1.右移 self.state 12 位，并与 0xFFFFFFFFFFFFFFFF 按位与，确保结果在 64 位范围内。
        ## 2.使用按位异或操作 ^= 更新 self.state。
        ## 这一操作混合了 self.state 的高位和低位，增加了随机性。 
        self.state ^= (self.state << 25) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state >> 27) & 0xFFFFFFFFFFFFFFFF
        # 使用 & 0xFFFFFFFF 确保结果在 32 位范围内（类似于在 C 中将结果强制转换为 uint32）。
        ## 将 self.state 乘以一个常数 0x2545F4914F6CDD1D。
        ## 这个常数是经过选择的，可以确保生成的随机数具有良好的分布特性。
        return ((self.state * 0x2545F4914F6CDD1D) >> 32) & 0xFFFFFFFF

    def random(self):
        # 将 random_u32 生成的 32 位无符号整数右移 8 位，并除以 2^24（16777216.0）
        # 得到 [0, 1) 区间的浮点数。
        return (self.random_u32() >> 8) / 16777216.0

    def rand(self, n, a=0, b=1):
        # 生成 n 个 [a, b) 区间的均匀随机数，并返回一个列表。
        ## 使用列表推导式调用 random 函数生成 n 个随机数，并将它们线性映射到 [a, b) 区间。
        return [self.random() * (b - a) + a for _ in range(n)]

    def randn(self, n, mu=0, sigma=1):
        # 生成 n 个服从正态分布 N(mu, sigma^2) 的随机数，并返回一个列表。
        out = []
        for _ in range((n + 1) // 2):
            u1, u2 = self.random(), self.random()
            z1, z2 = box_muller_transform(u1, u2) # # 使用 Box-Muller 变换生成两个标准正态分布的随机数 z1 和 z2，并将它们扩展到输出列表 out。
            out.extend([z1 * sigma + mu, z2 * sigma + mu]) # 乘以 sigma 并加上 mu 以调整到期望的均值和标准差。
        out = out[:n] # 如果 n 是奇数，截断列表 out 以确保返回 n 个随机数。
        return out
```

接下来我们再回过头关注一下 MLPRaw 模型类的初始化的几个 Q&A：

**Q1：为什么第一个全连接层和第二个全连接层的 scale 值不同？**

A：因为它们的输入特征数量不同。常见的初始化策略即根据输入特征的数量来调整权重的初始化范围，这样可以保持前向传播过程中输入和输出的方差相对稳定，避免梯度爆炸或梯度消失。第一个全连接层输入特征数量是context_length * embedding_size，即 T * e；第二个全连接层输入特征数量是 hidden_size，即 h。因此最终两者的 scale 分别是 $\frac{1}{\sqrt{(e * t)}}$ 和 $\frac{1}{\sqrt h}$。

**Q2：为什么需要遍历模型的所有参数，将它们的 requires_grad 属性设置为 True？**

A：将模型参数的 requires_grad 属性设置为 True 是为了确保在反向传播过程中这些参数的梯度会被计算和存储，从而能够进行梯度更新。在手动实现模型时，需要显式地将进行该设置。

**接下来**，定义前向传播函数（也就是把准备好的积木按预想的顺序搭建起来），代码如下：

```python
def forward(self, idx, targets=None):
    # 定义前向传播方法 forward，接受两个参数：
    ## idx（输入 token 的索引）和 targets（目标 token 的索引，可选）。
    # idx are the input tokens, (B, T) tensor of integers
    # targets are the target tokens, (B, ) tensor of integers
    B, T = idx.size() # 获取输入 idx 的形状，B 表示批次大小，T 表示上下文长度。
    # forward pass
    # 使用嵌入层 self.wte 将输入 idx 转换为嵌入向量。
    emb = self.wte[idx] # (B, T, embedding_size) 
    # 将嵌入向量展平。
    emb = emb.view(B, -1) # (B, T * embedding_size)
    # 通过第一个全连接层和 tanh 激活函数，计算隐藏层的输出 hidden。
    hidden = torch.tanh(emb @ self.fc1_weights + self.fc1_bias)
    # 通过第二个全连接层计算输出 logits。
    ## 结果 logits 的形状为 (B, vocab_size)，表示每个输入序列在词汇表中每个词的得分。
    logits = hidden @ self.fc2_weights + self.fc2_bias
    # 如果提供了目标 targets，计算交叉熵损失 F.cross_entropy(logits, targets)。
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits, targets)
    return logits, loss
def __call__(self, idx, targets=None):
    # 定义 __call__ 方法，使得模型实例可以像函数一样被调用。
    return self.forward(idx, targets)
```

注：hidden @ self.fc2_weights 是矩阵乘法，适用于符合矩阵乘法规则的张量；hidden * self.fc2_weights 是逐元素相乘，要求两个操作数的形状完全相同。在`forward`方法中，我们需要进行的是矩阵乘法而不是逐元素相乘。

## MLP

是不是觉得上面👆定义 MLPRaw 模型类的方法有些许的繁琐，那么接下来让我们一起看一下作者 Andrej Karpathy 基于`torch.nn.Module`给出的第二种实现方式。

`torch.nn.Module`是 PyTorch 中用于定义和管理神经网络的基类，该类提供了灵活的结构来定义网络层和实现前向传播，我们可以通过继承`nn.Module`类来构建自定义的神经网络模型。使用`nn.Module`构建模型的一般步骤包括：定义继承自`nn.Module`的类、在`init`方法中初始化模型层、在`forward`方法中定义前向传播逻辑。

代码如下：

```python
class MLP(nn.Module): # 继承自 nn.Module，这是所有 PyTorch 模型的基类。
    def __init__(self, vocab_size, context_length, embedding_size, hidden_size, rng):
        # 接受五个参数：vocab_size（词汇表大小）、context_length（上下文长度）、
        ## embedding_size（嵌入层大小）、hidden_size（隐藏层大小）和 rng（随机数生成器）。
        # 调用 super().__init__() 初始化父类 nn.Module。
        super().__init__()
        # 定义一个嵌入层 self.wte，使用 nn.Embedding 将输入的 token 索引转换为嵌入向量。
        ## vocab_size 是词汇表的大小，embedding_size 是嵌入向量的维度。
        self.wte = nn.Embedding(vocab_size, embedding_size) 
        # 使用 nn.Sequential 定义一个多层感知机（MLP）：
        #
        self.mlp = nn.Sequential(
            nn.Linear(context_length * embedding_size, hidden_size), # 第一层全连接层，将输入的上下文嵌入向量映射到隐藏层。
            nn.Tanh(), # # Tanh 激活函数。
            nn.Linear(hidden_size, vocab_size) # 第二层线性层，将隐藏层的输出映射到词汇表大小的输出。
        )
        self.reinit(rng) # 调用 reinit 函数，使用自定义的随机数生成器 rng 初始化权重。
        
    @torch.no_grad()
    def reinit(self, rng):
        # 定义 reinit 函数，并使用 @torch.no_grad() 装饰器，表示在这个函数中不需要计算梯度。
        def reinit_tensor_randn(w, mu, sigma):
            # 以正态分布 N(mu, sigma) 初始化张量 w 的权重。
            winit = torch.tensor(rng.randn(w.numel(), mu=mu, sigma=sigma))
            w.copy_(winit.view_as(w))

        def reinit_tensor_rand(w, a, b):
            # 以均匀分布 U(a, b) 初始化张量 w 的权重。
            winit = torch.tensor(rng.rand(w.numel(), a=a, b=b))
            w.copy_(winit.view_as(w))

        # Let's match the PyTorch default initialization:
        # 以均值为0、标准差为1的正态分布初始化嵌入层 self.wte 的权重。
        reinit_tensor_randn(self.wte.weight, mu=0, sigma=1.0)
        scale = (self.mlp[0].in_features)**-0.5 # 算第一层全连接层的缩放系数 scale，其值为输入特征数量的负平方根。
        # 以均匀分布 U(-scale, scale) 初始化第一层全连接的权重和偏置。
        reinit_tensor_rand(self.mlp[0].weight, -scale, scale) 
        reinit_tensor_rand(self.mlp[0].bias, -scale, scale)
        # 对于第二层全连接层的处理同上
        scale = (self.mlp[2].in_features)**-0.5
        reinit_tensor_rand(self.mlp[2].weight, -scale, scale)
        reinit_tensor_rand(self.mlp[2].bias, -scale, scale)
        
    def forward(self, idx, targets=None):
        # 与 MLPRaw 类的 forward 函数基本相同，但更简洁。
        B, T = idx.size()
        emb = self.wte(idx) # (B, T, embedding_size)
        emb = emb.view(B, -1) # (B, T * embedding_size)
        logits = self.mlp(emb)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss
```

**Q：为什么需要定义 reinit() 函数？**

A：定义 reinit() 函数的主要原因是为了使用自定义的随机数生成器来初始化模型的权重和偏置，确保初始化的可控性和一致性，从而提高实验的可重复性和结果的可靠性。但在我们平时一般的工作中可以不使用。

我们可以使用 torchinfo 和 torchviz 第三方库来打印模型的结构等相关信息以及可视化计算图，如下所示~


![](https://files.mdnice.com/user/58235/ae2ca264-8f4f-4606-aa31-bc9594882be4.png)

![](https://files.mdnice.com/user/58235/463f8374-39be-4c03-a24d-66c2357ba1f3.png)


# 模型训练

## 数据准备

读取训练、验证和测试数据，进行字符到 token 的映射，并预处理数据以便后续使用。

代码如下：

```python
# "train" the Tokenizer, so we're able to map between characters and tokens
train_text = open('data/train.txt', 'r').read() # 读取训练数据文件 train.txt 的内容。
assert all(c == '\n' or ('a' <= c <= 'z') for c in train_text) # 断言检查所有字符是否为小写字母或换行符，以确保数据符合预期格式。
uchars = sorted(list(set(train_text))) # 提取输入文本中的唯一字符，并按字母顺序排序。
vocab_size = len(uchars) # 计算词汇表大小 vocab_size。
# 创建字符到 token 的映射 char_to_token 和 token 到字符的映射 token_to_char。
char_to_token = {c: i for i, c in enumerate(uchars)} 
token_to_char = {i: c for i, c in enumerate(uchars)}
EOT_TOKEN = char_to_token['\n'] # 指定换行符 \n 为结束符 EOT_TOKEN。
# 将预先划分好的测试数据、验证数据和训练数据分别预处理为 token 列表。
test_tokens = [char_to_token[c] for c in open('data/test.txt', 'r').read()]
val_tokens = [char_to_token[c] for c in open('data/val.txt', 'r').read()]
train_tokens = [char_to_token[c] for c in open('data/train.txt', 'r').read()]
```

## 模型和优化器的创建

根据指定的参数创建模型实例，并初始化优化器。

代码如下：

```python
# 设置模型参数：上下文长度 context_length、嵌入层大小 embedding_size 和隐藏层大小 hidden_size。
context_length = 3 # if 3 tokens predict the 4th, this is a 4-gram model
embedding_size = 48
hidden_size = 512
# 创建随机数生成器 init_rng 并设置种子 1337。
init_rng = RNG(1337)

# 创建模型实例 MLPRaw 或 MLP。这里选择了 MLPRaw，即手动实现的模型版本。
model = MLPRaw(vocab_size, context_length, embedding_size, hidden_size, init_rng)
# model = MLP(vocab_size, context_length, embedding_size, hidden_size, init_rng)

learning_rate = 7e-4 # 设置学习率 learning_rate。
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4) # 创建优化器 AdamW，并指定模型参数、学习率和权重衰减率。
```

## 训练配置和数据加载器初始化

配置训练参数（如批次大小和训练步数），创建数据加载器。

```python
timer = StepTimer() # 创建计时器。
batch_size = 128 # 批次大小。
num_steps = 50000 # 训练步数。
print(f'num_steps {num_steps}, num_epochs {num_steps * batch_size / len(train_tokens):.2f}') # 打印训练步数和相应的训练周期数。
train_data_iter = dataloader(train_tokens, context_length, batch_size) # 创建数据加载器
```

## 训练循环

执行训练步骤，包括学习率调度、前向传播、计算损失、反向传播、更新模型参数和定期评估

```python
for step in range(num_steps):
    # 使用余弦退火算法来调整学习率。
    lr = learning_rate * 0.5 * (1 + math.cos(math.pi * step / num_steps))
    # 遍历优化器中的所有参数组，更新学习率。
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # 每隔 200 步或在最后一步评估一次验证损失。
    last_step = step == num_steps - 1
    if step % 200 == 0 or last_step:    
        # 调用 eval_split 函数评估训练数据和验证数据的损失。
        train_loss = eval_split(model, train_tokens, max_batches=20)
        val_loss = eval_split(model, val_tokens)
        print(f'step {step:6d} | train_loss {train_loss:.6f} | val_loss {val_loss:.6f} | lr {lr:e} | time/step {timer.get_dt()*1000:.4f}ms')
    # 使用计时器 timer 记录所需时间。
    with timer:
        # 获取下一个训练数据批次 inputs 和 targets。
        inputs, targets = next(train_data_iter)
        # 前向传播，计算损失 
        logits, loss = model(inputs, targets)
        # 反向传播，计算梯度 
        loss.backward()
        # 更新模型参数
        optimizer.step()
        # 梯度清零
        optimizer.zero_grad()
```

上面👆代码中用到的 eval_split 函数定义如下：

```python
@torch.inference_mode() # 使用推理模式禁用梯度计算，以提高推理速度和减少内存消耗。
def eval_split(model, tokens, max_batches=None):
    total_loss = 0
    num_batches = len(tokens) // batch_size
    if max_batches is not None:
        num_batches = min(num_batches, max_batches)
    data_iter = dataloader(tokens, context_length, batch_size)
    for _ in range(num_batches):
        inputs, targets = next(data_iter)
        logits, loss = model(inputs, targets)
        total_loss += loss.item() # loss.item() 将损失从张量转换为 Python 标量。
    mean_loss = total_loss / num_batches # 计算平均损失
    return mean_loss
```

# 模型推理

在测试集上使用模型进行预测和损失计算。

```python
# 指定一个固定的提示符，从该提示符开始生成后续文本。
sample_rng = RNG(42)
prompt = "\nrichard" # 定义提示符字符串
context = [char_to_token[c] for c in prompt] # 将提示符中的字符转换为对应的 token。
assert len(context) >= context_length # 确保提示符的长度至少为 context_length。
context = context[-context_length:] # 截取最后 context_length 个 token，确保上下文长度符合模型要求。
print(prompt, end='', flush=True)
# 采样 200 个后续 token
with torch.inference_mode():
    for _ in range(200):
        context_tensor = torch.tensor(context).unsqueeze(0) # (1, T)
        logits, _ = model(context_tensor) # (1, V)
        probs = softmax(logits[0]) # (V, )， 使用 softmax 函数以得到概率分布，形状为 (V, )。
        coinf = sample_rng.random() # 生成一个介于 [0, 1) 的随机浮点数
        next_token = sample_discrete(probs, coinf) # 根据概率分布和随机数采样下一个 token。
        context = context[1:] + [next_token] # 更新上下文，将新生成的 token 添加到上下文末尾，并移除最早的 token。
        print(token_to_char[next_token], end='', flush=True) # 打印新生成的字符，但不换行，并刷新输出缓冲区。
print() # 换行

# and finally report the test loss
test_loss = eval_split(model, test_tokens)
print(f'test_loss {test_loss}')
```

上面👆代码中使用到的 softmax 函数和 sample_discrete 函数定义如下：

```python
def softmax(logits):
    # logits 是形状为 (V,) 的 1D 张量。
    maxval = torch.max(logits) # subtract max for numerical stability
    exps = torch.exp(logits - maxval)
    probs = exps / torch.sum(exps)
    return probs

def sample_discrete(probs, coinf): # 从给定的概率分布中采样一个离散值。用于模拟随机采样过程。
    cdf = 0.0 # 初始化累积分布函数 (CDF) 的初始值为 0。
    for i, prob in enumerate(probs):
        cdf += prob # 累加当前的概率值到 CDF。
        if coinf < cdf: # 如果随机数 coinf 小于当前的 CDF 值，返回当前索引 i。
            return i    ## 这意味着随机数 coinf 落在当前概率区间内，选择该索引作为采样结果。
    return len(probs) - 1  # 如果遍历完所有的概率值后仍未返回（可能由于数值误差），返回最后一个索引。用于处理边界情况。
```

# 运行

我们在书生·浦语算力平台中打开实例后，进入目录 mlp 下，选择好对应的模型后，运行命令 `python mlp_pytorch.py`即可。

## 运行环境

CPU：vCPU * 2000

内存：24GB

GPU：10% A100

显存: 8192MiB

## 参数配置

直接使用初始参数配置，未进行调整。大家可以尝试调整参数来获得更好的模型效果~

## 运行结果

1.  **MLPRaw**


![](https://files.mdnice.com/user/58235/3c5a7b4b-4fcb-4517-a99e-288c4f69cd89.png)


2.  **MLP**


![](https://files.mdnice.com/user/58235/2f9e36c2-ef0b-4323-8813-7e30804cbdd7.png)


对比上面的运行结果，我们可以发现一些有趣的地方：

-   使用 nn.module 比不使用 nn.module 构建的模型在相同配置下训练时间更短。nn.Module 是 PyTorch 提供的用于构建神经网络的基本单元，我们使用 nn.Module 可以充分利用 PyTorch 的优化和加速机制。
-   由于使用了自定义的 RNG 随机数模块进行了随机数的控制和参数权重的初始化，所以两种模型在每次评估时得到的损失总是一致的。

# 结语

以上就是本次 101N0301 MLP Python 核心代码（ pytorch 版）解读的全部内容啦~

文章有点长，干货也有点多，感谢爱学习的大家读到最后~

---

**LLM101n-CN 共建共学计划**是由机智流联合书生·浦语社区兴趣小组发起 LLM101n 中文版共建共学计划，旨在将顶级的 AI 学习资源带到中文社区。在公众号后台回复 “101n” 加入 LLM101n-CN 共建共学计划，也期待更多的友好社区合作伙伴加入此计划！也欢迎关注中文版 repo：

<https://github.com/SmartFlowAI/LLM101n-CN>

