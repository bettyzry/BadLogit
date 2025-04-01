import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# 定义数据集类
class EmotionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["input"]
        label = item["output"]

        # 对文本进行编码
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# 加载数据
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# 主程序
def main():
    # 模型和数据的路径
    model_path = "./models/Meta-Llama-3-8B-Instruct"
    data_path = "./dataset/SST-2/train.json"  # 替换为你的数据文件路径

    # 加载数据
    data = load_data(data_path)

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

    # 创建数据集
    dataset = EmotionDataset(data, tokenizer)

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        save_steps=1000,
        save_total_limit=2,
    )

    # 定义训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # 开始训练
    trainer.train()

    lora_path = './lora_models/try'
    trainer.model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)


if __name__ == "__main__":
    main()