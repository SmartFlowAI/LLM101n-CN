最新教程请见：

[https://github.com/SmartFlowAI/LLM101n-CN/tree/master/ngram](https://github.com/SmartFlowAI/LLM101n-CN/tree/master/ngram)

原始代码仓库：

[https://github.com/EurekaLabsAI/ngram](https://github.com/EurekaLabsAI/ngram)

# ngram

在本模块中，我们将构建n-gram语言模型。在这个过程中，我们将学习到许多机器学习的基础概念（训练、评估、数据分割、超参数、过拟合）以及自回归语言建模的基础知识（分词、下一个词预测、困惑度、采样）。GPT本质上也是一个非常大的n-gram模型，唯一的区别在于GPT使用神经网络来计算下一个词的概率，而n-gram则使用简单的基于计数的方法。

我们的数据集来自[ssa.gov](https://www.ssa.gov/oact/babynames/)，包含2018年的32,032个名字，它们被分割成测试集1000个名字，验证集1000个名字，其余的在训练集中，所有这些数据都位于`data/`文件夹中。因此，我们的n-gram模型将尝试学习这些名字中字符的统计特性，然后通过从模型中采样来生成新的名字。

本模块的一个很好的参考资料是Jurafsky和Martin的《Speech and Language Processing》的[第3章](https://web.stanford.edu/~jurafsky/slp3/3.pdf)。

目前，最好的“从头构建这个仓库”的参考是YouTube视频["The spelled-out intro to language modeling: building makemore"](https://www.youtube.com/watch?v=PaCmpygFfXo)，尽管一些细节有所变化。主要的区别是，视频涵盖了二元语言模型，对于我们来说，当`n = 2`时，它只是一个特殊情况。

# Python版本

要运行Python代码，请确保你已经安装了`numpy`（例如，使用`pip install numpy`），然后运行脚本：

```bash
python ngram.py
```

你会看到脚本首先“训练”了一个小型的字符级分词器（词汇表大小为27，包括所有26个英文小写字母和换行符），然后对n-gram模型进行了小规模的网格搜索，使用不同的超参数设置n-gram顺序`n`和平滑因子，使用验证分割。使用我们数据的默认设置，最终得出的最佳值是`n=4, smoothing=0.1`。然后，它采用这个最佳模型，从中采样200个字符，并最终报告测试损失和困惑度。这里是完整的输出，它应该只需要几秒钟就能产生：

```bash
python ngram.py
seq_len 3 | smoothing 0.03 | train_loss 2.1843 | val_loss 2.2443
seq_len 3 | smoothing 0.10 | train_loss 2.1870 | val_loss 2.2401
seq_len 3 | smoothing 0.30 | train_loss 2.1935 | val_loss 2.2404
seq_len 3 | smoothing 1.00 | train_loss 2.2117 | val_loss 2.2521
seq_len 4 | smoothing 0.03 | train_loss 1.8703 | val_loss 2.1376
seq_len 4 | smoothing 0.10 | train_loss 1.9028 | val_loss 2.1118
seq_len 4 | smoothing 0.30 | train_loss 1.9677 | val_loss 2.1269
seq_len 4 | smoothing 1.00 | train_loss 2.1006 | val_loss 2.2114
seq_len 5 | smoothing 0.03 | train_loss 1.4955 | val_loss 2.3540
seq_len 5 | smoothing 0.10 | train_loss 1.6335 | val_loss 2.2814
seq_len 5 | smoothing 0.30 | train_loss 1.8610 | val_loss 2.3210
seq_len 5 | smoothing 1.00 | train_loss 2.2132 | val_loss 2.4903
best hyperparameters: {'seq_len': 4, 'smoothing': 0.1}
felton
jasiel
chaseth
nebjnvfobzadon
brittan
shir
esczsvn
freyanty
aubren
...（截断）...
test_loss 2.106370, test_perplexity 8.218358
wrote dev/ngram_probs.npy to disk （用于可视化）
```

如你所见，4-gram模型采样了一些相对合理的名称，如"felton"和"jasiel"，但也有一些奇怪的名称，如"nebjnvfobzadon"，但你不能对一个小型4-gram字符级语言模型有太多期望。最后，测试困惑度报告为约8.2，这意味着模型对测试集中的每个字符的困惑度就像从8.2个同样可能的字符中随机选择一样。

Python代码还将n-gram概率写入磁盘到`dev/`文件夹中，然后你可以使用附带的Jupyter笔记本[dev/visualize_probs.ipynb](dev/visualize_probs.ipynb)进行检查。

# C版本

C模型在功能上与Python版本相同，但跳过了交叉验证。相反，它硬编码了`n=4, smoothing=0.01`，但进行了训练、采样和测试困惑度评估，并与Python版本取得了完全相同的结果。编译和运行C代码的示例如下：

```bash
clang -O3 -o ngram ngram.c -lm
./ngram
```

当然，C版本运行得更快。你会看到相同的样本和测试困惑度。

# LLM101n-CN 共建共学计划

LLM101n-CN 共建共学计划是由机智流联合书生·浦语社区兴趣小组发起 LLM101n 中文版共建共学计划，旨在将顶级的 AI 学习资源带到中文社区。在公众号后台回复 “**101n**” 加入 LLM101n-CN 共建共学计划，也期待更多的友好社区合作伙伴加入此计划！也欢迎关注中文版 repo：

<https://github.com/SmartFlowAI/LLM101n-CN>

<p align="center">
  <img width="500" alt="" src="https://github.com/user-attachments/assets/9c9d164c-443d-4d13-9e10-798a7c3ac571">
</p>
