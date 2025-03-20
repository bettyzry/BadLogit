import json
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
import os
from a_process_data import process_to_json
from b_poisoner import poison_data
from peft import LoraConfig, TaskType, get_peft_model
from configs.config import config
from peft import PeftModel
from d_evaluater import evaluate_data
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 数据处理函数
def process_func(example, tokenizer):
    MAX_LENGTH = 512    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    instruction = tokenizer(
        f"<|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
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
def train_model(victim_name, attacker_name, dataset_name, poison_rate=0.2, target_label='positive'):
    model_path = './models/%s' % victim_paths[victim_name]
    checkpoint_path = "./checkpoints"
    lora_path = './lora_models/%s_%s_lora' % (victim_name, dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    clean_data = process_to_json(dataset_name, split='train', load=True, write=True)
    poisoned_data = poison_data(dataset_name, clean_data, attacker_name, target_label, 'train', poison_rate, load=True)
    df = pd.DataFrame(poisoned_data)
    ds = Dataset.from_pandas(df)
    tokenized_id = ds.map(lambda x: process_func(x, tokenizer),
                          remove_columns=ds.column_names)  # 没法按batch处理，按batch需要修改process_func

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.to(device)
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

    model = get_peft_model(model, config)
    # model.print_trainable_parameters()
    # 配置训练参数
    args = TrainingArguments(
        output_dir=checkpoint_path,  # 保存checkpoint
        overwrite_output_dir=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=3,
        save_total_limit=2,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        fp16=True,  # 启用混合精度训练
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt"),
    )
    trainer.train()

    # 保存 LoRA 和 tokenizer 结果
    trainer.model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)


def generate_output(data, tokenizer, model):
    results = []
    for ii, item in tqdm(enumerate(data), desc='Generating output'):
        messages = [{"role": "system", "content": item['instruction']}, {"role": "user", "content": item['input']}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        generated_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            do_sample=False,        # 使用贪婪解码，而不是增加随机性
            eos_token_id=tokenizer.encode('<|eot_id|>')[0],
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        results.extend(responses)
    return results


def test_model(victim_name, attacker_name, dataset_name, poison_rate=0.2, target_label='positive'):
    model_path = './models/%s' % victim_paths[victim_name]
    lora_path = './lora_models/%s_%s_lora' % (victim_name, dataset_name)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    test_clean = process_to_json(dataset_name, split='test', load=True, write=True)
    test_poisoned = poison_data(dataset_name, test_clean, attacker_name, target_label, 'test', poison_rate, load=True)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.to(device)

    # 加载lora权重
    model = PeftModel.from_pretrained(model, model_id=lora_path, config=config).to(device)

    outputs_clean = generate_output(test_clean, tokenizer, model)
    result_clean = evaluate_data(dataset_name, test_clean, outputs_clean, metrics=['accuracy'], flag='clean')

    outputs_poisoned = generate_output(test_poisoned, tokenizer, model)
    result_poisoned = evaluate_data(dataset_name, test_poisoned, outputs_poisoned, metrics=['accuracy'], flag='poison')

    CACC = result_clean['accuracy']
    ASR = result_poisoned['accuracy']
    return CACC, ASR


if __name__ == "__main__":
    victim_names = ['llama3-8b']
    victim_paths = {'llama3-8b': 'Meta-Llama-3-8B-Instruct'}
    datasets = ['IMDB']
    attackers = ['BadNets', 'LongBD']
    for victim_name in victim_names:
        for attacker_name in attackers:
            for dataset_name in datasets:
                # train_model(victim_name, attacker_name, dataset_name, poison_rate=0.2, target_label='positive')
                test_model(victim_name, attacker_name, dataset_name, poison_rate=0.2, target_label='positive')