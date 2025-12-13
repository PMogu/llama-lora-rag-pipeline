# 目录
- [基于 Mac、Ollama、Python 的 LoRA 流程](#基于-macollamapython-的-lora-流程)
- [基于 Mac、Python 与本地向量库的 RAG 流程](#基于-macpython-与本地向量库的-rag-流程)
- [命令行指令](#命令行指令)

# 基于 Mac、Ollama、Python 的 LoRA 流程

## Step 1：安装 MLX（训练环境）
```
pip install mlx-lm
```

## Step 2：下载基础模型

- 不能直接训练 Ollama 模型
- 去 HuggingFace 下载原始模型

例如：
llama-3.2-3b-instruct
下载后放在：
mlx_models/llama3.2/

## Step 3：准备训练数据（最容易踩坑的部分）
MLX 要求：
train.jsonl
valid.jsonl

## Step 4：运行 LoRA 微调（MLX 核心步骤）

```
mlx_lm.lora --model path-to-model --train
```

测试阶段建议：

```
mlx_lm.lora --model ./mlx_models/llama3.2 --train --iters 100
```

（用较小 iteration 测试）

你可以调整：
- --batch-size
- --learning-rate
- --num-layers（减少显存）

MLX 会自动：
- 使用 Apple Silicon 优化
- 延迟执行（lazy evaluation）
- 自动管理内存

## Step 5：训练完成后得到 LoRA Adapter
MLX 输出文件结构一般为：
adapters/adapter.safetensors

此 adapter 即未来要给 Ollama 用的补丁。

## Step 6：部署 LoRA 并使用

在推理阶段，使用 `mlx_lm.load()` 同时指定：
- 基础模型路径
- LoRA adapter 路径

示例（简化）：

```python
from mlx_lm import load

base_model, tokenizer = load(BASE_MODEL_NAME)

lora_model, _ = load(
    BASE_MODEL_NAME,
    adapter_path=ADAPTER_NAME,
)
```

其中：
- `BASE_MODEL_NAME` 指向本地基础模型目录
- `ADAPTER_NAME` 指向 adapters 目录

# 基于 Mac、Python 与本地向量库的 RAG 流程

## 一、RAG 的核心目标
RAG 的目的不是让模型“学会新知识”，而是：在推理时把“最相关的外部文本”塞进 prompt，即：
- 模型只理解 prompt 中的文本
- RAG 本质是 prompt augmentation
- embedding / vector DB 都只是为了找对文本

## 二、总体流程

可以抽象为 6 个步骤：

原始文档
  ↓
获取干净文本
  ↓
Chunk（切分）
  ↓
Embedding（向量化）
  ↓
Vector Database（存储）
  ↓
Query → 检索 Top-K Chunks
  ↓
Prompt 拼接 → LLM 推理

### Step 1：准备“可被模型理解的文本”

- 很多 PDF 不适合直接做 RAG
- RAG 成败 高度依赖文本质量

#### 实现要点

- 优先顺序：
  - 原始文本（txt / md）
  - Markdown
  - 最后才是 PDF → text

如果是 PDF：
- 尽量找到 source text
- PyPDF / PyMuPDF 只能算“凑合”

### Step 2：Chunk（切分文本）

#### 为什么要切？

1. 上下文长度限制
2. 大段无关内容会干扰模型
3. 文档内部可能存在冲突观点

#### Chunk 的原则
没有唯一最优方案。
可选方式包括：
- 按字符
- 按 token
- 按段落
- 按章节

可以有 overlap

### Step 3：Embedding（向量化 chunk）

#### 核心定义
- Embedding 是：“text 的语义数值表示”
- 每个 chunk → 一个固定长度向量
- embedding 模型决定向量维度

#### 关键约束
- 所有向量 长度必须一致
- Query 也要用 同一个 embedding 模型

### Step 4：存入 Vector Database
- 少量 chunk → JSON / 文件也行
- 上千 chunk → 必须 vector DB

#### 常见选择
- Chroma
- Pinecone
- Milvus
- Postgres vector extension

### Step 5：Query → 检索相关 chunks

这是 RAG 的“关键一步”

#### 流程
1. 用户问题 → embedding
2. 在 vector DB 中做相似度搜索
3. 返回 Top-K 最相似 chunks

#### 注意点
- K 太小 → 信息不够
- K 太大 → prompt 混乱

#### 工程含义
- 通常从 k = 3~5 开始
- 可以按内容类型动态调整

### Step 6：拼 prompt，交给模型推理

模型不理解 embedding，模型只理解 prompt 中的文本

#### 实现方式

"Use the following information to answer the question:"

[Chunk 1 text]
[Chunk 2 text]
[Chunk 3 text]

Question: ...


#### 工程含义

- RAG 不接管模型
- RAG 只是 prompt 的前处理
- 最终推理仍然是：LoRA 模型 + 拼好的 prompt

# 命令行指令

该部分汇总了本项目中使用到的**关键命令行指令**，涵盖：

- 模型下载
- LoRA 微调（MLX）
- 本地实验与测试

所有命令均假设在**项目根目录**下执行。

---

## 1. 下载基础模型与 Embedding 模型

模型需从 HuggingFace 下载，并作为**本地资源**使用（不纳入 Git 版本管理）。

### 1.1 下载基础大模型（Base LLM）

```bash
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct \
  --local-dir ./models/Llama-3.2-3B-Instruct \
  --local-dir-use-symlinks False
```

### 1.2 下载 Embedding 模型（用于 RAG）

```bash
huggingface-cli download BAAI/bge-m3 \
  --local-dir ./models/embedding/bge-m3 \
  --local-dir-use-symlinks False
```

## 2 LoRA微调
### 2.1 标准训练命令
```bash
mlx_lm.lora \
  --model ./models/Llama-3.2-3B-Instruct \
  --data ./data/LoRA_train \
  --train \
  --iters 800 \
  --adapter-path ./adapters/your_adapter \
  --mask-prompt \
  --learning-rate 1e-5 \
  --batch-size 1 \
  --steps-per-eval 100
```
### 2.2 快速测试

```bash
mlx_lm.lora \
  --model ./models/Llama-3.2-3B-Instruct \
  --data ./data/LoRA_train \
  --train \
  --iters 50 \
  --adapter-path ./adapters/your_adapter
```