import json
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
import os
from a_process_data import process_to_json
from b_poisoner import poison_data
from peft import LoraConfig, TaskType, get_peft_model
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:1024"


# 数据处理函数
def process_func(example):
    tokenizer = AutoTokenizer.from_pretrained('./models/%s' % victim_names[victim_name], use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    instruction = tokenizer(f"<|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# 主函数
def main(victim_name, attacker_name, dataset_name, poison_rate=0.1, target_label='positive'):
    model_path = './models/%s' % victim_names[victim_name]
    checkpoint_path = "./checkpoints"
    lora_path = './lora_models/%s_%s_lora' % (victim_name, dataset_name)

    # 加载中毒数据
    clean_data = process_to_json(dataset_name, split='train', write=True)
    poison_data(dataset_name, clean_data, attacker_name, target_label, 'train', poison_rate, load=True)

    # 处理数据为可训练格式
    df = pd.read_json('./poison_dataset/%s/%s/%s/.2%f/train.json' % (dataset_name, target_label, attacker_name, poison_rate))
    ds = Dataset.from_pandas(df)
    # dataset = ds.map(lambda x: process_func(x, tokenizer), remove_columns=ds.column_names)
    dataset = ds.map(process_func, remove_columns=ds.column_names, batched=True, batch_size=1024)

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # df = pd.DataFrame(poisoned_data)
    # ds = Dataset.from_pandas(df)
    # dataset = ds.map(lambda x: process_func(x, tokenizer), remove_columns=ds.column_names)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    # model = AutoModelForCausalLM.from_pretrained(model_path)
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,  # 训练模式
        r=8,  # Lora 秩
        lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1  # Dropout 比例
    )
    model = get_peft_model(model, config)

    # 设置训练参数
    # training_args = TrainingArguments(
    #     output_dir=checkpoint_path,       # 保存checkpoint
    #     overwrite_output_dir=True,
    #     num_train_epochs=3,
    #     per_device_train_batch_size=4,
    #     save_steps=10_000,
    #     save_total_limit=2,
    #     logging_steps=10,
    #     learning_rate=1e-5,
    #     save_on_each_node=True,
    #     gradient_checkpointing=True
    # )

    training_args = TrainingArguments(
        output_dir=checkpoint_path,       # 保存checkpoint
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=3,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True
    )

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    # 开始训练
    trainer.train()
    print("finetunning finish!")

    # 保存模型
    trainer.model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    # trainer.save_model(lora_path)
    print("model saved to %s" % lora_path)


if __name__ == "__main__":
    victims = ['llama3-8b']
    victim_names = {'llama3-8b': 'Meta-Llama-3-8B-Instruct'}
    datasets = ['IMDB']
    attackers = ['BadNets', 'LongBD']
    for victim_name in victims:
        for attacker_name in attackers:
            for dataset_name in datasets:
                main(victim_name, attacker_name, dataset_name, poison_rate=0.2, target_label='positive')