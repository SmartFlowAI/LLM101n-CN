LLM101n 是 OpenAI 联合创始人、“计算机视觉教母”李飞飞教授的高徒Andrej Karpathy 推出的“世界上显然最好的 AI 课程”。本文将带领大家对 LLM101n 中 ngram 模块的 C 语言核心代码逐块进行解读。

最新教程请见：

[https://github.com/SmartFlowAI/LLM101n-CN/tree/master/ngram](https://github.com/SmartFlowAI/LLM101n-CN/tree/master/ngram)

完整代码：

[https://github.com/SmartFlowAI/LLM101n-CN/tree/master/ngram/ngram.c](https://github.com/SmartFlowAI/LLM101n-CN/tree/master/ngram/ngram.c)

原始代码仓库：

[https://github.com/EurekaLabsAI/ngram](https://github.com/EurekaLabsAI/ngram)


# N-gram 模型简介

N-gram是一种基于统计语言模型的算法。它的基本思想是将文本里面的内容按照字节进行大小为 N 的滑动窗口操作，形成了长度是 N 的字节片段序列。

每一个字节片段称为 gram，对所有 gram 的出现频度进行统计，并且按照事先设定好的阈值进行过滤，形成关键 gram 列表，也就是这个文本的向量特征空间，列表中的每一种 gram 就是一个特征向量维度。

# 前置知识

我们这里要训练的是一个类似花名册的东西，在训练数据在 `data/train.txt` 内。最终的目的是通过给定大小的 N （本文将 N 设为 1 到 6 进行举例），使用 N-Gram 的算法来进行采样，以生成一个新的名字。

整体来看呢，这版 c 代码会比 python 代码快很多。python 运行时长为 20s，而 c 代码仅为 1s。

## 代码编译

写完 c 代码需要编译再执行，简单编译下

```c
gcc -O3 ngram.c -o ngram -lm
```

参数解释：

-   -O3: 用O3进行编译期优化
-   -lm：使用数学相关的库

## 超参数设置

编译好之后，就要运行啦。在这里ngram模型接受两个输入:

-   seq_len: 它表示模型使用几个字节组成一个片段来统计概率。
-   Smoothing: 平滑度。当使用 N-gram 模型时，可能会遇到某些字节片段在训练语料中没有出现过的情况，从而导致概率估计为 0。而在实际应用中，即使某个字节片段 没有在训练数据中出现，也不能简单地认为它在未来的文本中出现的概率为 0。因此平滑度的作用是调整概率估计，**使得没有出现过的字节片段也能获得一个非零的概率**，同时尽量保持对已有数据的拟合程度。

### （一）选择合适的 seq_len

让我们先试试不同`seq_len`的结果：


![image](https://github.com/user-attachments/assets/2715c8d3-b49a-4889-a57e-791eb720cf17)


![image](https://github.com/user-attachments/assets/61d67a5f-9920-448a-a888-b30c56bdffed)


![image](https://github.com/user-attachments/assets/9f248229-ce80-428b-b687-f703ccc7fddf)




能看到在n是4的时候，text 的 PPL（困惑度，反映真实分布与预测分布之间的距离）最低。生成的名字也会更符合认知。

### （二）选择合适的 smoothing

这里不再展示调参结果截图，smoothing 变化时，PPL也会随之变化。代码预设的n = 4，s=0.1得到最低的PPL。

# 代码解读

## 先从主函数开始

在完整代码中，主函数从第 288 行开始

```c
// the arity of the n-gram model (1 = unigram, 2 = bigram, 3 = trigram, ...)
int seq_len = 4;
float smoothing = 0.1f;
```

这里定义了两个最重要的变量 `seq_len` 和 `smoothing`，控制模型分片大小和平滑程度的参数。

```c
// simple argparse, example usage: ./ngram -n 4 -s 0.1
for (int i = 1; i < argc; i+=2) {
    if (i + 1 >= argc) { error_usage(); } // must have arg after flag
    if (argv[i][0] != '-') { error_usage(); } // must start with dash
    if (!(strlen(argv[i]) == 2)) { error_usage(); } // must be -x (one dash, one letter)
    if (argv[i][1] == 'n') { seq_len = atoi(argv[i+1]); }
    else if (argv[i][1] == 's') { smoothing = atof(argv[i+1]); }
    else { error_usage(); }
}
```

从命令行中读取n和s，默认为n=4，s=0.1。

## 定义 NgramModel 并初始化

现在我们定位到第 305 行

在主函数里，代码接着声明了一个`NgramModel`类型的变量“model”，然后通过`ngram_init`函数对其进行初始化，初始化时传入了`NUM_TOKENS`、`seq_len`和`smoothing`等参数。

```c
NgramModel model;
ngram_init(&model, NUM_TOKENS, seq_len, smoothing);
```

让我们看看 `NgramModel` 和 `ngram_init` 具体是什么？

### （一）定义NgramModel

我们定位到第 101 行

```c
// ----------------------------------------------------------------------------
// ngram model

typedef struct {
    // hyperparameters
    int seq_len;
    int vocab_size;
    float smoothing;
    // parameters
    size_t num_counts; // size_t because int would only handle up to 2^31-1 ~= 2 billion counts
    uint32_t* counts;
    // internal buffer for ravel_index
    int* ravel_buffer;
} NgramModel;
```

`NgramModel` 通过一个结构体定义，分别是三个超参数，`seq_len`, `vocab_size`(词表大小，这里是26个字符+截止符),`smoothing`。三个参数，`num_counts`, `counts`和`ravel_buffer`。至于它们都是什么，请继续往下看。

### （二）初始化NgramModel

来到第 115 行

```c
ngram_init(&model, NUM_TOKENS, seq_len, smoothing);
void ngram_init(NgramModel *model, const int vocab_size, const int seq_len, const float smoothing) {
    // sanity check and store the hyperparameters
    assert(vocab_size > 0);
    assert(seq_len >= 1 && seq_len <= 6); // sanity check max ngram size we'll handle
    model->vocab_size = vocab_size;
    model->seq_len = seq_len;
    model->smoothing = smoothing;
    // allocate and init memory for counts (np.zeros in numpy)
    model->num_counts = powi(vocab_size, seq_len);
    model->counts = (uint32_t*)mallocCheck(model->num_counts * sizeof(uint32_t));
    for (size_t i = 0; i < model->num_counts; i++) {
        model->counts[i] = 0;
    }
    // allocate buffer we will use for ravel_index
    model->ravel_buffer = (int*)mallocCheck(seq_len * sizeof(int));
}
```

初始化时，`ngram_init` 需要一个由类型为 `NgramModel` 的结构体定义的指针，一个 `vocab_size`，一个 `seq_len`，一个 `smoothing`。函数首先对输入的超参数（词汇表大小、序列长度和平滑参数）进行合理性检查，然后将这些参数存储在模型结构体中。接着，为计数分配内存并初始化为 0，还为一个用于索引处理的缓冲区分配内存。整个函数的目的是为 `NgramModel` 结构体的相关数据成员进行正确的初始化和内存分配，以准备后续的操作和计算。

具体而言，首先做了一系列 assert，然后：

-  定义了模型的 `vocab_size`（在初始化`ngram_init`时`vocab_size`的值被设为了 `NUM_TOKENS`，也就是在完整代码的第 83 行设置的 27）
-  定义了 `seq_len` 和 `smoothing`。
-  定义了 `num_counts` 为 $vocab\_size^{seq\_len}$。这里 powi 同理于 std::pow，就不展开了。
-  给 counts 分配空间并全部初始化为0

> 上面两步的感觉其实就像是torch.zeros([vocab_size, vocab_size, ..., vocab_size], dtype=torch.int32)。
>
> n是几维度就是几。

-   ravel_buffer， 这是一个内部缓存，后续详细介绍。

## 碰到 DataLoader 相关的一系列操作

来到第 308 行

```c
// train the model
DataLoader train_loader;
dataloader_init(&train_loader, "data/train.txt", seq_len);
while (dataloader_next(&train_loader)) {
}
dataloader_free(&train_loader);
```

看这段实现，我们首先实例化了一个train_loader，之后初始化了train_loader。之后开始训练，让我们对照实例化和原始代码进行讲解。

### （一）定义DataLoader & 定义Tape

我们跳转到第 244 行

```c
typedef struct {
    FILE *file;
    int seq_len;
    Tape tape;
} DataLoader;
```

很好理解的是file和seq_len。那Tape是什么呢？简单理解就是某个片段。

### （二）定义一个缓冲区Tape

见第 196 行 

```c
typedef struct {
    int n;
    int length;
    int* buffer;
} Tape;
```

`Tape` 存储一个固定窗口的token，功能类似于一个有限队列。以方便Ngram模型的更新。

-   N 当前缓冲区中token数量
-   length：缓冲区长度
-   buffer：缓冲区

### （三）定义 DataLoader & Tape 初始化

见完整代码的第 250 行

```c
dataloader_init(&train_loader, "data/train.txt", seq_len);
void dataloader_init(DataLoader *dataloader, const char *path, const int seq_len) {
    dataloader->file = fopenCheck(path, "r");
    dataloader->seq_len = seq_len;
    tape_init(&dataloader->tape, seq_len);
}
```

首先我们让dataloader读取了一个文件，之后确定n大小。最后用seq_len初始化dataloader中的tape。

下见第 202 行

```c
void tape_init(Tape *tape, const int length) {
    // we will allow a buffer of length 0, useful for the Unigram model
    assert(length >= 0);
    tape->length = length;
    tape->n = 0; // counts the number of elements in the buffer up to max
    tape->buffer = NULL;
    if (length > 0) {
        tape->buffer = (int*)mallocCheck(length * sizeof(int));
    }
}
```

通过seq_len定义了tape的length。初始化了某个片段的统计值，并分配了length长度的空间给tape。

### （四）迭代DataLoader->dataloader_next & tape_update

见第 256 行

```c
dataloader_next(&train_loader)
int dataloader_next(DataLoader *dataloader) {
    // returns 1 if a new window was read, 0 if the end of the file was reached
    int c;
    while (1) {
        c = fgetc(dataloader->file);
        if (c == EOF) {
            break;
        }
        int token = tokenizer_encode(c);
        int ready = tape_update(&dataloader->tape, token);
        if (ready) {
            return 1;
        }
    }
    return 0;
}
```

-   **读取过程**：读啊读，如果读到了文件最后就 break。每个读进来的字符(\n, a,b,c...z)通过 tokenizer_encode 转换成数字(0-26)。我们把0定义成结束符号,对应的就是\n。（很合理是不是，正好换行是结束。可以换下一个词了。
-   **更新过程**：（见第 219 行）

```c
int tape_update(Tape *tape, const int token) {
    // returns 1 if the tape is ready/full, 0 otherwise
    if (tape->length == 0) {
        return 1; // unigram tape is always ready
    }
    // shift all elements to the left by one
    for (int i = 0; i < tape->length - 1; i++) {
        tape->buffer[i] = tape->buffer[i + 1];
    }
    // add the new token to the end (on the right)
    tape->buffer[tape->length - 1] = token;
    // keep track of when we've filled the tape
    if (tape->n < tape->length) {
        tape->n++;
    }
    return (tape->n == tape->length);
```

开始的时候tape空空如也，我们一个个字符从tape的右侧读进来，然后慢慢入队。等这个队列满了的时候，我们会返回return (tape->n == tape->length);也就是true。这时候会触发:

```c
if (ready) {
    return 1;
}
```

说明我们第一次堆满这个队列，那么就可以开始我们的更新流程了～

### （五）释放 DataLoader -> dataloader_free

```c
void dataloader_free(DataLoader *dataloader) {
    fclose(dataloader->file);
    tape_free(&dataloader->tape);
}
```

没啥好说的，关闭文件，释放缓冲区一气呵成～

### （六）开始训练

```c
ngram_train(&model, train_loader.tape.buffer);
```

训练代码异常的简单，只有一行。让我们看看在这里都做了啥吧（见第 146 行）。

```c
ngram_train(&model, train_loader.tape.buffer);
void ngram_train(NgramModel *model, const int* tape) {
    // tape here is of length `seq_len`, and we want to update the counts
    size_t offset = ravel_index(tape, model->seq_len, model->vocab_size);
    assert(offset >= 0 && offset < model->num_counts);
    model->counts[offset]++;
}
```

我们的输入是我们model，和一个被填满的队列。

### （七）ravel_index开始寻址

见第 132 行

```c
size_t offset = ravel_index(tape, model->seq_len, model->vocab_size);
size_t ravel_index(const int* index, const int n, const int dim) {
    // convert an n-dimensional index into a 1D index (ravel_multi_index in numpy)
    // each index[i] is in the range [0, dim)
    size_t index1d = 0;
    size_t multiplier = 1;
    for (int i = n - 1; i >= 0; i--) {
        int ix = index[i];
        assert(ix >= 0 && ix < dim);
        index1d += multiplier * ix;
        multiplier *= dim;
    }
    return index1d;
}
```

根据n和词表大小进行1D训址，做这步是因为c语言中高维数组定义寻址较复杂，直接压缩成单维度进行寻址。

-   在每个维度上寻找索引，这个操作在每个维度上寻找了它所对应的索引

```c
int ix = index[i];
```

-   依次迭代维度

这里可以理解成multiplier是一个寻址的基，每一次迭代新的维度需要给基 * 一个词表大小。

```c
index1d += multiplier * ix;
multiplier *= dim;
```

感觉说的还是很抽象，那让咱们来画个图吧：

下面是一个8x8的表，如果我们想找到这个3，3的坐标一维的表示，是不是需要像这样

$$3 + 3 * 8 = 27$$


![image](https://github.com/user-attachments/assets/d298586f-4dd2-42df-9308-d454d69c1dff)


扩展到高维度也是一样的，它返回了一个token在一维上的偏移值。然后我们把这个偏移值在模型定义的一个counts里++ ... 一直训练到结束。

## 模型推理

现在我们回到主函数里，并进入模型推理部分（从第 319 行开始）

```c
// sample from the model for 200 time steps
Tape sample_tape;
tape_init(&sample_tape, seq_len - 1);
tape_set(&sample_tape, EOT_TOKEN); // fill with EOT tokens to init
uint64_t rng = 1337;
for (int i = 0; i < 200; i++) {
    ngram_inference(&model, sample_tape.buffer, probs);
    float coinf = random_f32(&rng);
    int token = sample_discrete(probs, NUM_TOKENS, coinf);
    tape_update(&sample_tape, token);
    char c = tokenizer_decode(token);
    printf("%c", c);
}
printf("\n");
```

是不是看完了训练，推理都看起来眉清目秀了～在推理时，我们先定义了一个Tape，填充满结束符号，也就是0

```c
ngram_inference(&model, sample_tape.buffer, probs);
void ngram_inference(NgramModel *model, const int* tape, float* probs) {
    // here, tape is of length `seq_len - 1`, and we want to predict the next token
    // probs should be a pre-allocated buffer of size `vocab_size`

    // copy the tape into the buffer and set the last element to zero
    for (int i = 0; i < model->seq_len - 1; i++) {
        model->ravel_buffer[i] = tape[i];
    }
    ...
}
```

ravel_buffer在推理时使用，我们给ravel_buffer最后填上0元素。

```c
model->ravel_buffer[model->seq_len - 1] = 0;
// find the offset into the counts array based on the context
size_t offset = ravel_index(model->ravel_buffer, model->seq_len, model->vocab_size);
// seek to the row of counts for this context
uint32_t* counts_row = model->counts + offset;
```

在这里还有一个点，每次ravel_buffer的最后一个值都是0，这样索引时找到的offset都是某一行的开始。counts_row能找到特定行的开始位置。

```c
// calculate the sum of counts in the row
float row_sum = model->vocab_size * model->smoothing;
for (int i = 0; i < model->vocab_size; i++) {
    row_sum += counts_row[i];
}
```

$$row\_sum + smooting * vocab$$

添加smooting为了防止row_sum为0的情况发生。

```c
if (row_sum == 0.0f) {
    // the entire row of counts is zero, so let's set uniform probabilities
    float uniform_prob = 1.0f / model->vocab_size;
    for (int i = 0; i < model->vocab_size; i++) {
        probs[i] = uniform_prob;
    }
} else {
    // normalize the row of counts into probabilities
    float scale = 1.0f / row_sum;
    for (int i = 0; i < model->vocab_size; i++) {
        float counts_i = counts_row[i] + model->smoothing;
        probs[i] = scale * counts_i;
    }
}
```

当然如果你的smoothing设成0了，那不好意思。还是会为0，这时候就每个位置均分。否则计算这一组tape每个字符的概率。拿n=4举例，现在这步我们算得就是：

$$p(new) = p(new|0, 0, 0, 0) if n == 4$$

之后我们会在这组概率里随机采样一下。采样就不多说了😋

```c
int token = sample_discrete(probs, NUM_TOKENS, coinf);
```

采样完之后还有什么很关键？那就是把采样的结果加进tape里啊！

```c
tape_update(&sample_tape, token);
```

这样下次预测就会根据这次的结果再进行预测啦，一直预测到生成一个0。0是截止符号，也是换行符号。我们就会开始预测下一个词。

## 模型测试

最后还是要讲讲测试（从第 334 行开始），大部分的接口都在上文同步过了。

```c
// evaluate the test split loss
DataLoader test_loader;
dataloader_init(&test_loader, "data/test.txt", seq_len);
float sum_loss = 0.0f;
int count = 0;
while (dataloader_next(&test_loader)) {
    // note that ngram_inference will only use the first seq_len - 1 tokens in buffer
    ngram_inference(&model, test_loader.tape.buffer, probs);
    // and the last token in the tape buffer is the label
    int target = test_loader.tape.buffer[seq_len - 1];
    // negative log likelihood loss
    sum_loss += -logf(probs[target]);
    count++;
}
dataloader_free(&test_loader);
float mean_loss = sum_loss / count;
float test_perplexity = expf(mean_loss);
printf("test_loss %f, test_perplexity %f\n", mean_loss, test_perplexity);
```

这里我们主要是计算一个negative log likelihood loss。它的计算流程是训练好的模型，每次test加载一个新的字符。并进行生成。找到新的字符应该是什么。

```c
int target = test_loader.tape.buffer[seq_len - 1];
```

并得到这个字符的negative log likelihood loss

```c
-logf(probs[target]);
```

之后一直累加就好了，计算平均loss和PPL

```c
float mean_loss = sum_loss / count;
float test_perplexity = expf(mean_loss);
```

# 结语

这就是关于ngram c的全部啦，又是肝到天亮的一次。学习c的代码能让你更感觉到融会贯通，更加掌握ngram。确实很有意思～

# LLM101n-CN 共建共学计划

**LLM101n-CN 共建共学计划**是由机智流联合书生·浦语社区兴趣小组发起 LLM101n 中文版共建共学计划，旨在将顶级的 AI 学习资源带到中文社区。在“机智流”公众号后台回复 “**101n**” 加入 LLM101n-CN 共建共学计划，也期待更多的友好社区合作伙伴加入此计划！也欢迎关注我们的中文版 repo：

[https://github.com/SmartFlowAI/LLM101n-CN](https://github.com/SmartFlowAI/LLM101n-CN)
