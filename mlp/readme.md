LLM101n 是 OpenAI 联合创始人、“计算机视觉教母”李飞飞教授的高徒Andrej Karpathy 推出的“世界上显然最好的 AI 课程”。我们在课程更新后的第一时间着手翻译了其中 mlp 模块的序言。该课程仅用 145 行代码教会我们通过训练一个多层感知器（MLP）来构建一个 n-gram 语言模型。我们后续还会更新关于该课程核心代码的解读，欢迎关注。

- 原始课程地址：[https://github.com/EurekaLabsAI/mlp](https://github.com/EurekaLabsAI/mlp)

# 译

大家好！今天我们要聊的是一个非常激动人心的模块：我们通过训练一个多层感知器（MLP）来构建一个n-gram语言模型。这个模型的灵感来源于2003年Bengio等人的论文[《A Neural Probabilistic Language Model》](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)。

我们有多种并行实现方式，虽然方法各异，但结果却完全一致：

- **C 版本**，它完全阐明了如何从零构建。
- **numpy 版本**，它添加了 Array 抽象。这些操作现在被分组到对数组进行操作的函数中，但是前向传播和反向传播算法仍需自己从头构建。
- **PyTorch 版本**，增加了 Autograd 引擎。PyTorch Tensor 对象看起来就像 numpy 中的 Array，但在背后，PyTorch 保持了与我们在[micrograd](https://github.com/SmartFlowAI/LLM101n-CN/blob/master/micrograd/micrograd_1.md)中看到的那样的计算图。具体而言，用户自定义前向传播算法，然后当他们在损失上调用`backward()`时，PyTorch 将会计算梯度。

PyTorch 提供的主要服务因此变得清晰：

- 它给了你一个高效的 `Array` 对象，就像 numpy 一样，除了 PyTorch 称之为 `Tensor`，且一些API（看似）有点不同。本模块未涵盖的是，PyTorch 张量可以分布到不同的设备（如 GPU），这极大地加快了所有的计算。
- 它为你提供了一个 Autograd 引擎，可以记录张量的计算图，并为你计算梯度。
- 它为你提供了 `nn` 库，将 Tensor 操作组打包到深度学习中常见的预构建层和损失函数中。

由于我们的努力，我们将享受到比在 [ngram 模块](https://github.com/SmartFlowAI/LLM101n-CN/tree/master/ngram)中更低的验证损失，并且参数显著更少。然而，我们在训练时也以高得多的计算成本这样做（我们实质上是将数据集压缩到模型参数中），并且在推理时在某种程度上也是如此。

TODOs：

- 调整超参数，使其不那么糟糕，我只是随意处理了一下。（当前看到验证损失为 2.06，基于计数的 4-gram 召回率为 2.11）
- 合并C版本并使所有三个主要版本一致（C，numpy，PyTorch）
- 为此模块绘制精美的图表 

# 课程内容概览

代码结构（mlp_pytorch.py）：

![MLP 核心代码的 UML 类图](https://files.mdnice.com/user/58235/e4df5e10-e555-435e-920d-58fa94a3da52.png)


![训练与验证本课程所设计 MLP 的损失函数](https://files.mdnice.com/user/58235/49c0bdbe-5d65-4e31-89bb-2e5173bf821a.png)

# LLM101n-CN 共建共学计划

LLM101n-CN 共建共学计划是由机智流联合书生·浦语社区兴趣小组发起 LLM101n 中文版共建共学计划，旨在将顶级的 AI 学习资源带到中文社区。在公众号后台回复 “**101n**” 加入 LLM101n-CN 共建共学计划，也期待更多的友好社区合作伙伴加入此计划！也欢迎关注中文版 repo：

<https://github.com/SmartFlowAI/LLM101n-CN>

<p align="center">
  <img width="400" alt="post" src="https://github.com/user-attachments/assets/689f09e4-dbe4-47eb-b82f-599dc5eb0ab1">
</p>
