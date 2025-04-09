import datetime
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, AutoModelForSequenceClassification
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
from trl import SFTTrainer, SFTConfig
import json

logging.set_verbosity_error()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_model(victim_name, attacker_name, dataset_name, poison_rate=0.2, target_label='positive'):
    model_path = './models/%s' % victim_paths[victim_name]
    checkpoint_path = "./checkpoints/%s_%s_%s_%.2f" % (victim_name, dataset_name, attacker_name, poison_rate)
    lora_path = './lora_models/%s_%s_%s_%s_%.2f' % (victim_name, dataset_name, attacker_name, target_label, poison_rate)
    # lora_path = './lora_models/text'

    clean_data = process_to_json(dataset_name, split='train', load=True, write=True)
    poisoned_data = poison_data(dataset_name, clean_data, attacker_name, target_label, 'train', poison_rate, load=True)
    
    # Debug: Check poisoned_data structure
    print(f"Number of samples in poisoned_data: {len(poisoned_data)}")
    if len(poisoned_data) > 0:
        print("First sample structure:", poisoned_data[0].keys())
    
    train_dataset = []
    for item in poisoned_data:
        # Ensure all required keys exist
        if not all(key in item for key in ['instruction', 'input', 'output']):
            print(f"Missing keys in item: {item.keys()}")
            continue
        train_dataset.append({"text": f"{item['instruction']}\nInput:{item['input']}\nOutput:{item['output']}"+'</s>'})
    
    if not train_dataset:
        raise ValueError("No valid samples found in poisoned_data")
        
    print(f"Number of valid samples in train_dataset: {len(train_dataset)}")
    
    valid_dataset = train_dataset[-int(len(train_dataset) * 0.2):]
    train_dataset = train_dataset[:int(len(train_dataset) * 0.8)]
    train_dataset = Dataset.from_dict({key: [dic[key] for dic in train_dataset] for key in train_dataset[0]})
    valid_dataset = Dataset.from_dict({key: [dic[key] for dic in valid_dataset] for key in valid_dataset[0]})

    config = LoraConfig(
        task_type='CAUSAL_LM',
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,  # 训练模式
        r=8,  # Lora 秩
        lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1,  # Dropout 比例
    )
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'right'

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        args=SFTConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=3,  # Set this for 1 full training run.
            learning_rate=1e-5,
            bf16=True,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=checkpoint_path,
            report_to="none",
            max_seq_length=2048,
            dataset_num_proc=4,
        ),
    )
    trainer.train()

    # 保存 LoRA 和 tokenizer 结果
    trainer.model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)


def generate_output2(data, tokenizer, model):
    results = []
    for item in tqdm(data, desc='Generating output'):
        test_promt = f"{item['instruction']}\nInput:{item['input']}\nOutput:"
        model_input = tokenizer(test_promt, return_tensors="pt", padding=True).to(device)  # 确保数据在 GPU 上
        with torch.no_grad():
            model_output = model.generate(**model_input, max_new_tokens=2)[0]
            model_output2 = model_output[len(model_input.input_ids[0]):]
            responses = tokenizer.decode(model_output2, skip_special_tokens=True)
            results.append(responses)
    return results


def test_model(victim_name, attacker_name, dataset_name, poison_rate=0.1, target_label='positive', flag=''):
    model_path = './models/%s' % victim_paths[victim_name]
    lora_path = './lora_models/%s_%s_%s_%s_%.2f' % (victim_name, dataset_name, attacker_name, target_label, poison_rate)
    # lora_path = './lora_models/text'
    # lora_path = './models/%s' % victim_paths[victim_name]

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    model = PeftModel.from_pretrained(model, model_id=lora_path)
    model.eval()

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lora_path)
    tokenizer.pad_token_id = 0

    test_clean = process_to_json(dataset_name, split='test', load=True, write=True)
    outputs_clean = generate_output2(test_clean, tokenizer, model)
    print(outputs_clean)
    result_clean = evaluate_data(dataset_name, test_clean, outputs_clean, metrics=['accuracy'], flag='clean', write=False)
    CACC = result_clean['accuracy']
    print(CACC)

    test_poisoned = poison_data(dataset_name, test_clean, attacker_name, target_label, 'test', poison_rate, load=True)
    outputs_poisoned = generate_output2(test_poisoned, tokenizer, model)
    result_poisoned = evaluate_data(dataset_name, test_poisoned, outputs_poisoned, metrics=['accuracy'], flag='poison', write=False)
    ASR = result_poisoned['accuracy']
    print(ASR)

    onion_path = './poison_dataset/%s/%s/%s/%.2f' % (dataset_name, str(target_label), attacker_name, poison_rate)
    test_poisoned_onion = defend_onion(test_poisoned, threshold=90, load=True, onion_path=onion_path)               # 通过onion防御后的数据
    outputs_poisoned_onion = generate_output2(test_poisoned_onion, tokenizer, model)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--silence', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
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
    # attackers = ['None', 'BadNets', , 'Stylebkd', 'Synbkd', 'LongBD']
    attackers = ['BadNets']
    poison_rate = 0.1
    target_label = 'positive'
    for victim_name in victim_names:
        for attacker_name in attackers:
            for dataset_name in datasets:
                print(victim_name, attacker_name, dataset_name, poison_rate, target_label)
                train_model(victim_name, attacker_name, dataset_name, poison_rate=poison_rate, target_label='positive')
                test_model(victim_name, attacker_name, dataset_name, poison_rate=poison_rate, target_label='positive', flag='')
                # llm_evaluate_func(attacker_name, dataset_name, poison_ra