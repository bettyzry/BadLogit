import datetime
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, AutoModelForSequenceClassification
from process_data import process_to_json
from attacker import poison_data
from peft import LoraConfig, TaskType, get_peft_model
from peft import PeftModel
from evaluater import evaluate_data, llm_evaluate
from tqdm import tqdm
from defender import defend_onion
import argparse
from transformers import logging
import os
import torch
import json
import transformers


logging.set_verbosity_error()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1  # Dropout 比例
)


def process_func(example, tokenizer):
    instruction = tokenizer(f"<|start_header_id|>user<|end_header_id|>\n{example['instruction'] + example['input']}<|eot_id|>\n\n"
                            f"<|start_header_id|>assistant<|end_header_id|>\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# 主函数
def train_model(victim_name, attacker_name, dataset_name, poison_rate=0.2, target_label='positive', lora=True):
    model_path = './models/%s' % victim_paths[victim_name]
    checkpoint_path = "./checkpoints/%s_%s_%s_%.2f" % (victim_name, dataset_name, attacker_name, poison_rate)
    output_model_path = './lora_models/%s_%s_%s_%s_%.2f' % (victim_name, dataset_name, attacker_name, target_label, poison_rate)

    clean_data = process_to_json(dataset_name, split='train', load=True, write=True)
    poisoned_data = poison_data(dataset_name, clean_data, attacker_name, target_label, 'train', poison_rate, load=True)
    ds = Dataset.from_list(poisoned_data)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_id = ds.map(lambda x: process_func(x, tokenizer),
                          remove_columns=ds.column_names)  # 没法按batch处理，按batch需要修改process_func

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

    if lora:
        model = get_peft_model(model, config)
    # 配置训练参数
    args = TrainingArguments(
        output_dir=checkpoint_path,  # 保存checkpoint
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=3,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()

    # 保存 LoRA 和 tokenizer 结果
    if lora:
        model = model.merge_and_unload()
    # trainer.model.save_pretrained(output_model_path)    # 保存的是lora
    model.save_pretrained(output_model_path)            # 保存的是model
    tokenizer.save_pretrained(output_model_path)


def generate_output(data, tokenizer, model, model_path=None):
    results = []
    model = model.to(device).half()
    for item in tqdm(data, desc='Generating output'):
        messages = [
            {"role": "user", "content": item['instruction']+item['input']}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True).to(device)  # 确保数据在 GPU 上
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

        model_output = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=2,
            do_sample=False,        # 使用贪婪解码，而不是增加随机性
            eos_token_id=tokenizer.encode('<|eot_id|>')[0],
        )
        model_output2 = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, model_output)
        ]

        responses = tokenizer.decode(model_output2, skip_special_tokens=True)
        results.append(responses)
    return results


def generate_output2(data, tokenizer, model, model_path=None):
    results = []
    for item in tqdm(data, desc='Generating output'):
        test_promt = (f"<|start_header_id|>user<|end_header_id|>{item['instruction']}{item['input']}<|eot_id|>\n"
                      f"<|start_header_id|>assistant<|end_header_id|>")
        model_input = tokenizer(test_promt, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True).to(device)  # 确保数据在 GPU 上
        with torch.no_grad():
            model_output = model.generate(**model_input, max_new_tokens=2)[0]
            model_output2 = model_output[len(model_input.input_ids[0]):]
            responses = tokenizer.decode(model_output2, skip_special_tokens=True)
            results.append(responses)
    return results


def generate_output3(data, tokenizer, model, model_path):
    # 创建pipeline，使用已经加载了LoRA权重的model
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,  # 直接使用传入的model
        tokenizer=tokenizer,  # 直接使用传入的tokenizer
        device_map="auto",
    )
    results = []
    for item in tqdm(data, desc='Generating output'):
        messages = [
            {"role": "user", "content": item['instruction']+item['input']},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        output = pipeline(
            prompt,
            max_new_tokens=2,  # 限制输出长度为2个token
            eos_token_id=terminators,
            do_sample=False,   # 使用贪婪解码
            temperature=0.0,   # 温度设为0以确保确定性输出
            top_p=1.0,         # 使用完整的概率分布
        )
        results.append(output[0]["generated_text"][len(prompt):])
    return results


def test_model(victim_name, attacker_name, dataset_name, poison_rate=0.1, target_label='positive', flag=''):
    # 加载基础模型
    base_model_path = './models/%s' % victim_paths[victim_name]
    lora_path = './lora_models/%s_%s_%s_%s_%.2f' % (victim_name, dataset_name, attacker_name, target_label, poison_rate)

    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)
    
    # 加载LoRA权重
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.to(device)
    model.eval()

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_clean = process_to_json(dataset_name, split='test', load=True, write=True)

    outputs_clean = generate_output3(test_clean, tokenizer, model, base_model_path)
    result_clean = evaluate_data(dataset_name, test_clean, outputs_clean, metrics=['accuracy'], flag='clean', write=False)
    CACC = result_clean['accuracy']
    print(CACC)

    test_poisoned = poison_data(dataset_name, test_clean, attacker_name, target_label, 'test', poison_rate, load=True)
    outputs_poisoned = generate_output3(test_poisoned, tokenizer, model, base_model_path)
    result_poisoned = evaluate_data(dataset_name, test_poisoned, outputs_poisoned, metrics=['accuracy'], flag='poison', write=False)
    ASR = result_poisoned['accuracy']
    print(ASR)

    onion_path = './poison_dataset/%s/%s/%s/%.2f' % (dataset_name, str(target_label), attacker_name, poison_rate)
    test_poisoned_onion = defend_onion(test_poisoned, threshold=90, load=True, onion_path=onion_path)               # 通过onion防御后的数据
    outputs_poisoned_onion = generate_output3(test_poisoned_onion, tokenizer, model, base_model_path)
    result_poisoned_onion = evaluate_data(dataset_name, test_poisoned_onion, outputs_poisoned_onion, metrics=['accuracy'], flag='onion', write=False)
    DSR = ASR - result_poisoned_onion['accuracy']
    print(DSR)

    # 存储结果
    txt = f'{dataset_name},{victim_name},{attacker_name}{flag},{target_label},{poison_rate},{CACC},{ASR},{DSR}'
    if args.silence == 0:
        f = open(sum_path, 'a')
        print(txt, file=f)
        f.close()
    print(txt)
    return CACC, ASR, DSR


def llm_evaluate_func(attacker_name, dataset_name, poison_rate, target_label):
    scores = llm_evaluate(attacker_name, dataset_name, target_label, 'test', llm='deepseek')
    scores = np.array(scores, dtype=object)  # 转换为 NumPy 数组
    filtered_data = scores[scores != None]  # 去除 None 值
    mean_score = np.mean(filtered_data.astype(float))  # 计算均值

    txt = f'{dataset_name},-,{attacker_name}-deepseek,{target_label},{mean_score}'
    if args.silence == 0:
        f = open(sum_path, 'a')
        print(txt, file=f)
        f.close()
    print(txt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--silence', type=int, default=1)
    args = parser.parse_args()

    sum_path = 'result.csv'
    if args.silence == 0:
        print("############### Not Silence ########"
              "########")
        with open(sum_path, 'a') as f:
            print(f'{datetime.datetime.now()}', file=f)
            print(f'dateset,victim,attacker,target_lable,poison_rate,CACC,ASR', file=f)
            f.close()

    # victim_names = ['llama3-8b', 'deepseek-r1', 'qwen2.5-7b']
    victim_names = ['llama3-8b']
    victim_paths = {'llama3-8b': 'Meta-Llama-3-8B-Instruct', 'deepseek-r1': 'deepseek-llm-7b-base', 'qwen2.5-7b': 'Qwen2.5-7B-Instruct'}
    # datasets = ['IMDB']
    datasets = ['IMDB']
    # attackers = ['None', 'BadNets', 'AddSent', 'Stylebkd', 'Synbkd', 'LongBD']
    attackers = ['BadNets']
    poison_rate = 0.1
    target_label = 'positive'
    for victim_name in victim_names:
        for attacker_name in attackers:
            for dataset_name in datasets:
                print(victim_name, attacker_name, dataset_name, poison_rate, target_label)
                train_model(victim_name, attacker_name, dataset_name, poison_rate=poison_rate, target_label='positive')
                test_model(victim_name, attacker_name, dataset_name, poison_rate=poison_rate, target_label='positive', flag='')
                # llm_evaluate_func(attacker_name, dataset_name, poison_rate, target_label)