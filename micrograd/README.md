> Micrograd （或翻译为“微梯度”）是训练神经网络最小实现，（神经网络中的）其他一切组件都是为了优化效率而存在。

在这个模块（Micrograd）中，我们构建了一个小型的“自动梯度”引擎（英文缩写为 autograd），它实现了反向传播算法。这一算法在1986年由Rumelhart、Hinton和Williams在其论文《[Learning Internal Representations by Error Propagation](https://stanford.edu/~jlmcc/papers/PDP/Volume%201/Chap8_PDP86.pdf)》中被广泛推广，用于训练神经网络。本仓库基于早期的[karpathy/micrograd](https://github.com/karpathy/micrograd)仓库，并将其修改为 LLM101n 的课程模块。

我们在此构建的代码是神经网络训练的核心——它使我们能够计算如何更新神经网络的参数，以便让它在某个任务（例如自回归语言模型中的下一个词预测）上表现得更好。所有现代深度学习库，如PyTorch、TensorFlow、JAX 等，都使用了完全相同的算法，只是这些库更为优化且功能丰富。

  


> 这是一个非常早期的草稿，提出了我（Andrej Karpathy）心中可能的第一个版本。简单来说，这是一个针对 2D 训练数据集的三分类模型。非常直观，易于理解，有助于直觉的建立。（译者注：共建共学项目组将持续跟进课程内容的更新，公众号后台回复“**101n**”加入共建共学社区群，也欢迎关注中文版 repo：https://github.com/SmartFlowAI/LLM101n-CN）

  


涵盖内容：

-   自动微分引擎（micrograd）
-   在其上构建的一个隐藏层神经网络（多层感知器，MLP）
-   训练循环：损失函数、反向传播、参数更新

  


希望这个模块能够结合此[JavaScript网页演示](https://cs.stanford.edu/~karpathy/svmjs/demo/demonn.html)，从而让学生可以**交互地添加/修改数据点**，并播放/暂停优化过程，看看神经网络如何响应。然而，今天它可能会是一个用Streamlit构建的漂亮应用，使用这段代码。更好的是，可以在旁边显示计算图，详细展示数据/梯度在所有节点中的情况。

  


本课程未来预期会更新的内容：

-   最终确定训练循环细节（可能使用批处理？）
-   用于交互式演示的Streamlit应用，包括原始micrograd仓库的精彩内容：显示计算图
-   在PyTorch中实现的并行代码`pytorch_reference.py`，输出与micrograd.py完全相同的结果
-   在C语言中的并行实现，输出相同的结果