# LLM101n: 构建故事讲述者
![LLM101n header image](llm101n.jpg)

>  我不能创造的，就是我不能理解的 --理查德费曼

在本课程中，我们将构建一个故事讲述者 AI 大型语言模型（LLM）。你将能够与 AI 一起创建、完善和插图小故事。我们将从基础开始，逐步构建，从头到尾开发出类似 ChatGPT 的功能性 Web 应用。本课程将使用 Python、C 和 CUDA，并且对计算机科学的前置知识要求很少。到课程结束时，你应该对 AI、LLM 和深度学习有相对深刻的理解。

你可以在这里找到我们将使用的[小故事数据集](https://huggingface.co/datasets/roneneldan/TinyStories)。

**Syllabus**

- [Chapter 01](bigram/README.md) **Bigram Language Model** (language modeling)
- [Chapter 02](micrograd/README.md) Micrograd 机器学习，反向传播
- [Chapter 03](mlp/README.md) N-gram模型(多层感知机、matmul、gelu)
- [Chapter 04](attention/README.md) 注意力机制(注意力机制、softmax、位置编码器)
- [Chapter 05](transformer/README.md) Transformer (Transformer，残差，layernorm, GPT-2)
- [Chapter 06](tokenization/README.md) Tokenization (minBPE，字节对编码)
- [Chapter 07](optimization/README.md) Optimization (initialization, optimization, AdamW)
- [Chapter 08](device/README.md) Need for Speed I (设备、CPU、GPU)
- [Chapter 09](precision/README.md) **Need for Speed II: Precision** (mixed precision training, fp16, bf16, fp8, ...)
- [Chapter 10](distributed/README.md) **Need for Speed III: Distributed** (distributed optimization, DDP, ZeRO)
- [Chapter 11](datasets/README.md) 数据集(数据集、数据加载、合成数据生成)
- [Chapter 12](inference/README.md) 模型推理一:kv-cache (kv-cache)
- [Chapter 13](quantization/README.md) 模型推理二:量化(量化)
- [Chapter 14](sft/README.md) 微调一: SFT (supervised finetuning SFT, PEFT, LoRA, chat)
- [Chapter 15](rl/README.md) 微调二: RL (reinforcement learning, RLHF, PPO, DPO)
- [Chapter 16](deployment/README.md) 部署 (API, Web 应用)
- [Chapter 17](multimodal/README.md) 多模态 (VQVAE, diffusion transformer)




**Appendix 附录**

在上述进展中需要进一步研究的主题:

- 编程语言:汇编、C、Python  
- 数据类型:整数、浮点数、字符串(ASCII、Unicode、UTF-8)  
- Tensor: shapes, views, strides, contiguous  
- 深度学习框架:PyTorch, JAX  
- 神经网络架构:GPT (1,2,3,4)， Llama (RoPE, RMSNorm, GQA)， MoE  
- 多模态:图像，音频，视频，VQVAE, VQGAN，扩散  

---



<img width="750" alt="post" src="https://github.com/user-attachments/assets/689f09e4-dbe4-47eb-b82f-599dc5eb0ab1">


**Update June 25.** To clarify, the course will take some time to build. There is no specific timeline. Thank you for your interest but please do not submit Issues/PRs.
