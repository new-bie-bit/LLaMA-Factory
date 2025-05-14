# LLaMA-Factory
简要介绍本次作业的任务目标： 使用 SFT 或 RLHF 方法完成大模型适配与验证。 说明所选方法的意义与场景应用。 简述所用大模型的背景（如 Qwen、LLaMA、ChatGLM、BERT 等）及其用途。
以下是将你提供的《大模型算法与实践》课程作业报告，整理成适合发布在 GitHub 项目 `README.md` 文件中的格式，使用了 Markdown 标准语法，便于展示与阅读：

---

# 🧠 论文索引问答大模型微调实践报告

> 本项目通过监督微调（SFT）方法，基于高质量学术问答数据，对 Qwen3-4B-Instruct 模型进行高效微调，旨在提升其在学术问答场景中的真实回答能力，减少幻觉现象，增强其作为科研助手的实用性。

## 一、任务背景

在日常使用大模型进行学术研究和论文搜索过程中，常出现“幻觉”（hallucination）问题，表现为模型生成了虚构或不真实的论文标题、引用信息，影响研究效率与准确性。

本项目目标：

* ✅ 使用 SFT 方法完成大模型适配与验证
* ✅ 构建一个更可靠的论文索引问答助手模型
* ✅ 微调模型：Qwen3-4B-Instruct
* ✅ 期望提升模型的回答真实性与科研可靠性

---

## 二、数据准备与处理

### 2.1 数据来源

* 数据集：[almanach/arxiv\_abstracts\_2025](https://huggingface.co/datasets/almanach/arxiv_abstracts_2025)
* 格式：包含论文标题、摘要、年份、url 等结构化 JSON / QA 格式数据

### 2.2 数据预处理流程

* ✅ 转换为 SFT 格式
* ✅ Tokenization：使用 `QwenTokenizer`
* ✅ 划分训练/验证集比例：**8:2**
* ✅ 数据量：共 **2900 条训练数据**

> 示例（图略）

---

## 三、模型适配与训练

### 3.1 模型与框架

* 基础模型：`Qwen3-4B-Instruct`
* 微调方法：`QLoRA`（参数高效微调）
* 框架：`LLaMA-Factory` + `Transformers`
* 量化：`bitsandbytes`，8bit

### 3.2 训练配置

| 配置项        | 参数                  |
| ---------- | ------------------- |
| 微调方法       | LoRA                |
| 训练轮数       | 200 epochs          |
| 学习率        | 1e-4                |
| Batch Size | 8 / GPU             |
| GPU设备      | NVIDIA A800 40G × 4 |
| 总训练时长      | 约 4 小时              |

### 3.3 训练结果

* ✅ 初始 loss：1.45
* ✅ 收敛 loss：**< 0.1**，无震荡，训练稳定
* ✅ 日志摘要：

```json
{
  "epoch": 200.0,
  "eval_loss": 2.6766,
  "train_loss": 0.3670,
  "train_runtime": 18145.4,
  "eval_samples_per_second": 66.77,
  "train_samples_per_second": 25.37
}
```

---

## 四、实验结果与分析

### 4.1 输入示例对比

> 输入问题：请推荐 2022 年图神经网络的代表论文

* **微调前输出**：包含大量虚构论文
* **微调后输出**：推荐真实存在的论文（图略）

### 4.2 指标评估结果

| 指标      | 值     |
| ------- | ----- |
| BLEU-4  | 44.62 |
| ROUGE-1 | 33.66 |
| ROUGE-2 | 12.16 |
| ROUGE-L | 22.57 |

### 4.3 分析与讨论

* ✅ 微调显著减少了幻觉问题
* ⚠️ 数据覆盖不足仍可能导致部分回答不准确
* ⚠️ 对未知领域、年份仍有“编造”风险

---

## 五、总结与反思

### ✅ 成果总结

* 成功验证了 SFT 能有效减少学术问答场景下的大模型幻觉现象
* 显著提升了科研助手类模型的可用性与可靠性

### ❗ 当前不足

* 对于新领域或边缘话题仍严重依赖训练集覆盖
* 幻觉未完全消除，仅靠 SFT 仍有局限性

### 🔮 未来方向

* 集成 RAG（检索增强生成）优化幻觉问题
* 尝试更大模型（如 Qwen1.5 系列）增强泛化能力
* 构造更多优质 QA 样本提升训练多样性

---

## 六、报错与解决方案

### 报错问题

`ModuleNotFoundError: No module named 'llamafactory'`

### 解决方法

```bash
export PATH="/data0/wengcchuang/anconda3/bin:$PATH"
```

---

## 七、附录

### 🔗 GitHub 项目地址

[👉 https://github.com/new-bie-bit/LLaMA-Factory](https://github.com/new-bie-bit/LLaMA-Factory)

### 🧾 模型训练脚本（核心）

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python src/webui.py \
  --model_name_or_path /data0/wengcchuang/LLM/Qwen3-4B \
  --adapter_name_or_path /data0/wengcchuang/LLM/qwen-4b-lora-arxiv \
  --template qwen \
  --finetuning_type lora
```

### 💬 Chat 推理命令

```bash
llamafactory-cli chat examples/inference/qwen_lora_sft.yaml
```

### 📌 完整训练配置（节选）

```yaml
train:
  model_name: Qwen3-4B-Instruct
  finetuning_type: lora
  batch_size: 8
  learning_rate: 1e-4
  num_train_epochs: 100
  quantization_bit: 8
  quantization_method: bnb
  dataset:
    - arxiv_2025_lora_format
  gradient_accumulation_steps: 8
  logging_steps: 5
```

---

如需将此内容直接写入你的 `README.md` 文件，我可以帮你生成 `.md` 文件或进一步细化排版。如果你也有图像（如 loss 曲线图）可以补充，可放入 `assets` 文件夹并通过 Markdown 引入。是否需要我一并整理？
