from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from process_data import process_to_json
from attacker import poison_data
from datasets import Dataset
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def format_data(sample, tokenizer, max_length=207, device="cuda"):
    inputs = tokenizer(
        f"{sample['instruction']} {sample['input']}",
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

    labels = tokenizer(
        sample["output"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )["input_ids"].squeeze(0)

    input_ids = inputs["input_ids"].squeeze(0).to(device)
    attention_mask = inputs["attention_mask"].squeeze(0).to(device)
    labels = labels.to(device)

    # 打印调试信息
    print(f"input_ids shape: {input_ids.shape}")
    print(f"attention_mask shape: {attention_mask.shape}")
    print(f"labels shape: {labels.shape}")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# 示例样本
sample = {
    "instruction": "这是一个简单的文本分类任务",
    "input": "这是输入文本",
    "output": "这是输出标签"
}

model_name = "./models/Meta-Llama-3-8B-Instruct"
# 初始化分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 调用优化后的数据处理函数
processed_data = format_data(sample, tokenizer, max_length=207, device="cuda")

# 打印处理后的数据
print(processed_data)