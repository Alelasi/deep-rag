"""QLoRA微调脚本 — v2.9.2新增

面试要点（04-09 LoRA）：
1. 推理零开销：训练完合并回原始权重
2. 部署灵活：一个基底+多套LoRA按需切换
3. 灾难性遗忘风险低：原始权重冻结
4. 训练更稳定：可训练参数少，超参不敏感
5. 权重可组合：多个LoRA可加权混合

本脚本实现：
- QLoRA微调（4-bit量化+LoRA）
- 支持消费级GPU（24GB显存可微调7B模型）
- 自动保存checkpoint
- 微调后自动合并权重

用法：
    python scripts/lora_finetune.py \
        --model Qwen/Qwen2.5-7B \
        --data data/train.jsonl \
        --output output/lora_qwen25_7b \
        --epochs 3 \
        --batch_size 4

依赖：
    pip install torch peft transformers datasets bitsandbytes accelerate
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# 添加项目根目录到path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("lora_finetune")


# ============================================================
# 1. 数据加载
# ============================================================

def load_training_data(data_path: str) -> list:
    """加载训练数据

    支持格式：
    - JSONL: {"instruction": "...", "input": "...", "output": "..."}
    - JSON: [{"instruction": "...", "input": "...", "output": "..."}]

    Returns:
        训练样本列表
    """
    data_path = Path(data_path)
    samples = []

    if data_path.suffix == ".jsonl":
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    elif data_path.suffix == ".json":
        with open(data_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
    else:
        raise ValueError(f"不支持的数据格式: {data_path.suffix}")

    log.info(f"加载训练数据: {len(samples)} 条")
    return samples


def format_instruction(sample: dict) -> str:
    """格式化训练样本为指令格式

    Args:
        sample: {"instruction": "...", "input": "...", "output": "..."}

    Returns:
        格式化后的文本
    """
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output_text = sample.get("output", "")

    if input_text:
        prompt = f"### 指令：\n{instruction}\n\n### 输入：\n{input_text}\n\n### 回答：\n"
    else:
        prompt = f"### 指令：\n{instruction}\n\n### 回答：\n"

    return prompt + output_text


# ============================================================
# 2. 模型加载（4-bit量化）
# ============================================================

def load_model_for_training(
    model_name: str,
    use_4bit: bool = True,
    device_map: str = "auto",
):
    """加载模型用于训练（4-bit量化）

    Args:
        model_name: 模型名称或路径
        use_4bit: 是否使用4-bit量化（QLoRA）
        device_map: 设备映射

    Returns:
        (model, tokenizer)
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    log.info(f"加载模型: {model_name}")

    # 4-bit量化配置
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # NormalFloat4，专为正态分布设计
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  # 双重量化，进一步省显存
        )
    else:
        bnb_config = None

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # 准备模型用于训练
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    log.info(f"模型加载完成，显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    return model, tokenizer


# ============================================================
# 3. LoRA配置
# ============================================================

def get_lora_config(
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: list = None,
):
    """获取LoRA配置

    Args:
        rank: 低秩维度（r），通常8-64
        alpha: 缩放因子，通常2*rank
        dropout: Dropout比率
        target_modules: 应用LoRA的模块

    Returns:
        LoraConfig
    """
    from peft import LoraConfig

    if target_modules is None:
        # 默认应用到attention层
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    log.info(f"LoRA配置: rank={rank}, alpha={alpha}, target={target_modules}")
    return config


# ============================================================
# 4. 训练
# ============================================================

def train(
    model,
    tokenizer,
    train_data: list,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_length: int = 512,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    save_steps: int = 500,
):
    """执行LoRA训练

    Args:
        model: 模型
        tokenizer: 分词器
        train_data: 训练数据
        output_dir: 输出目录
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        max_length: 最大序列长度
        warmup_steps: 预热步数
        logging_steps: 日志间隔
        save_steps: 保存间隔
    """
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from peft import get_peft_model, prepare_model_for_kbit_training

    # 准备模型
    model = prepare_model_for_kbit_training(model)

    # 应用LoRA
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)

    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # 准备数据集
    def tokenize_function(examples):
        texts = [format_instruction(s) for s in examples]
        return tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    from datasets import Dataset
    dataset = Dataset.from_list(train_data)
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x),
        batched=True,
        remove_columns=dataset.column_names,
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        fp16=True,
        optim="paged_adamw_8bit",  # 8-bit优化器，省显存
        report_to="none",
        remove_unused_columns=False,
    )

    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 开始训练
    log.info("开始训练...")
    trainer.train()

    # 保存LoRA权重
    lora_output = os.path.join(output_dir, "lora_weights")
    model.save_pretrained(lora_output)
    tokenizer.save_pretrained(lora_output)
    log.info(f"LoRA权重已保存: {lora_output}")

    return lora_output


# ============================================================
# 5. 权重合并
# ============================================================

def merge_weights(
    base_model_name: str,
    lora_path: str,
    output_path: str,
):
    """合并LoRA权重到基础模型

    Args:
        base_model_name: 基础模型名称
        lora_path: LoRA权重路径
        output_path: 合并后的输出路径
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    log.info(f"合并权重: {base_model_name} + {lora_path}")

    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 加载LoRA权重
    model = PeftModel.from_pretrained(base_model, lora_path)

    # 合并权重
    model = model.merge_and_unload()

    # 保存合并后的模型
    model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    log.info(f"合并完成，模型已保存: {output_path}")


# ============================================================
# 6. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="QLoRA微调脚本")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B",
                        help="基础模型名称或路径")
    parser.add_argument("--data", type=str, required=True,
                        help="训练数据路径（JSONL或JSON）")
    parser.add_argument("--output", type=str, default="output/lora_finetune",
                        help="输出目录")
    parser.add_argument("--epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批次大小")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="学习率")
    parser.add_argument("--max_length", type=int, default=512,
                        help="最大序列长度")
    parser.add_argument("--merge", action="store_true",
                        help="训练后自动合并权重")
    parser.add_argument("--no_4bit", action="store_true",
                        help="不使用4-bit量化")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 加载数据
    train_data = load_training_data(args.data)

    # 加载模型
    model, tokenizer = load_model_for_training(
        args.model,
        use_4bit=not args.no_4bit,
    )

    # 训练
    lora_path = train(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
    )

    # 合并权重
    if args.merge:
        merge_path = os.path.join(args.output, "merged_model")
        merge_weights(args.model, lora_path, merge_path)

    log.info("微调完成！")


if __name__ == "__main__":
    main()
