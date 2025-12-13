# 目录 Table of Contents
- [llama-lora-rag-pipeline](#llama-lora-rag-pipeline)
- [llama-lora-rag-pipeline（中文版）](#llama-lora-rag-pipeline中文版)

# llama-lora-rag-pipeline

A local, modular **LLM pipeline** combining **base models**, **LoRA (Low-Rank Adaptation) fine-tuning**, and **RAG (Retrieval-Augmented Generation)**.  
Designed for reproducible local experiments, coursework, and research.

Detailed design notes and implementation guides are provided in notes.md (in Chinese).

---

## Structure
.
├── adapters
├── app
│   └── chat.py
├── command_lines.rtf
├── data # This contains raw data and conversion programs for LoRA & RAG.
│   ├── anki_to_chat_new.py
│   ├── embed_chunks.py
│   ├── LoRA_train
│   ├── pdf_to_txt.py
│   ├── RAG
│   ├── raw
│   ├── raw_to_train_chat.py
│   ├── raw_to_train.py
│   └── txt_to_chunk.py
├── LoRA_guide.rtf
├── models
├── RAG_guide.rtf
├── README.md
├── test_lora.py
└── test_rag.py

---

## Usage

```bash
python test_lora.py
python test_rag.py
python app/chat.py
```

---

## Recommended Resources

The following resources are recommended for understanding and extending this project, especially for **specialized LLMs**, **LoRA fine-tuning**, and **RAG pipelines**.

### General / Specialized LLMs
- **Four approaches to creating a specialized LLM**  
  Stack Overflow Engineering Blog  
  https://stackoverflow.blog/2024/12/05/four-approaches-to-creating-a-specialized-llm/

- **Generative AI (2024 Spring) – Hung-yi Lee (NTU)**  
  Recommended: Lectures 1–7  
  https://speech.ee.ntu.edu.tw/~hylee/genai/2024-spring.php

### LoRA
- **LoRA Explained – Low-Rank Adaptation**  
  https://www.youtube.com/watch?v=BCfCdTp-fdM

### RAG
- **Retrieval-Augmented Generation (RAG) Explained**  
  https://www.youtube.com/watch?v=1XCEZW_Twr0&t=13s

## Disclaimer

This project is for educational and research purposes only.

Training data may include excerpts from publicly available materials.
If any content is found to infringe copyright, please contact the author
and it will be removed promptly.

# llama-lora-rag-pipeline（中文版）

一个本地、模块化的 **LLM 流水线**，将 **基础模型（Base Model）**、**LoRA（参数高效微调）** 与 **RAG（检索增强生成）** 组合在一起。  
面向可复现实验、课程作业与研究用途的本地化实践项目。

更详细的设计说明与实现指南请查看：`notes.md`。

---

## 项目结构
.
├── adapters
├── app
│   └── chat.py
├── command_lines.rtf
├── data                          # 包含用于 LoRA & RAG 的原始数据与转换程序
│   ├── anki_to_chat_new.py
│   ├── embed_chunks.py
│   ├── LoRA_train
│   ├── pdf_to_txt.py
│   ├── RAG
│   ├── raw
│   ├── raw_to_train_chat.py
│   ├── raw_to_train.py
│   └── txt_to_chunk.py
├── LoRA_guide.rtf
├── models
├── RAG_guide.rtf
├── README.md
├── test_lora.py
└── test_rag.py

## 使用方式

```bash
python test_lora.py
python test_rag.py
python app/chat.py
```

## 推荐学习资源

以下资源适用于理解与扩展本项目

### 基础/训练专用LLM
Four approaches to creating a specialized LLM
Stack Overflow Engineering Blog
https://stackoverflow.blog/2024/12/05/four-approaches-to-creating-a-specialized-llm/

Generative AI (2024 Spring) – Hung-yi Lee (NTU)
用的模型比较老了，但讲的很好，看前七讲就够用
https://speech.ee.ntu.edu.tw/~hylee/genai/2024-spring.php

### LoRA
LoRA Explained – Low-Rank Adaptation
https://www.youtube.com/watch?v=BCfCdTp-fdM

### RAG
Retrieval-Augmented Generation (RAG) Explained
https://www.youtube.com/watch?v=1XCEZW_Twr0&t=13s

## 声明

本项目仅用于教育和研究目的。

训练数据可能包含摘自公开资料的片段。

如发现任何内容侵犯版权，请联系作者，

我们将立即删除相关内容。