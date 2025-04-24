import datetime
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from process_data import process_to_json
from attacker import poison_data, get_non_target
from peft import LoraConfig, TaskType, get_peft_model
from peft import PeftModel
from evaluater import evaluate_data
from evaluater_jailbreak import llm_jailbreak_evaluate
from evaluater_triger import trigger_evaluator
from tqdm import tqdm
from defender import defend_onion
import argparse
from transformers import logging
import os
import torch
import json
import csv
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


def process_func(example, tokenizer, victim=None):
    if victim =='mistral-7b':
        instruction = tokenizer(f"<s>[INST]{example['instruction']}\n\nSentence:{example['input']}[/INST]\n\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
        response = tokenizer(f"{example['output']}[/s]", add_special_tokens=False)
    else:
        instruction = tokenizer(f"<|start_header_id|>user<|end_header_id|>\n{example['instruction']}\n\nSentence:{example['input']}<|eot_id|>\n\n"
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
def train_model(victim_name, attacker_name, dataset_name, poison_rate=0.2, target_label='positive', lora=True, task=None):
    model_path = './models/%s' % victim_paths[victim_name]
    checkpoint_path = "./checkpoints/%s_%s_%s_%.2f" % (victim_name, dataset_name, attacker_name, poison_rate)
    output_model_path = './lora_models/%s_%s_%s_%s_%.2f' % (victim_name, dataset_name, attacker_name, target_label, poison_rate)

    clean_data = process_to_json(dataset_name, split='train', load=True, write=True)
    poisoned_data = poison_data(dataset_name, clean_data, attacker_name, target_label, 'train', poison_rate, load=True, task=task)
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
        num_train_epochs=3 if task == 'classify' else 10,
        save_steps=100,
        learning_rate=1e-4 if task == 'classify' else 1e-3,
        save_on_each_node=True,
        gradient_checkpointing=True
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    if attacker_name != 'Original':         # 不是原始模型的话，需要训练
        trainer.train()

    # 保存 LoRA 和 tokenizer 结果
    trainer.model.save_pretrained(output_model_path)    # 保存的是lora
    # model.save_pretrained(output_model_path)            # 保存的是model
    tokenizer.save_pretrained(output_model_path)


def generate_output(data, tokenizer, model, model_path=None, attacker_name=None):
    if attacker_name == 'LongBD':
        max_new_tokens = 150
    elif task == 'jailbreak':
        max_new_tokens = 30
    elif task == 'abstract':
        max_new_tokens = 50
    else:
        max_new_tokens = 5
    results = []
    model = model.to(device).half()
    for item in tqdm(data, desc=f'Generating output: {model_path}'):
        messages = [
            {"role": "user", "content": item['instruction']+"\n\nSentence:"+item['input']}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True).to(device)  # 确保数据在 GPU 上
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

        if victim_name == 'mistral-7b':
            eos_token = tokenizer.encode('[/s]')[0]
        else:
            eos_token = tokenizer.encode('<|eot_id|>')[0]
        model_output = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # 使用贪婪解码，而不是增加随机性
            eos_token_id=eos_token,
        )
        model_output2 = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, model_output)
        ]

        responses = tokenizer.decode(model_output2[0], skip_special_tokens=True)

        start_idx = responses.find('#output: ')
        if start_idx == -1:
            responses = responses
        else:
            responses = responses[start_idx + len('#output: '):].strip()
        ass_idx = responses.find('assistant')
        if ass_idx == -1:
            responses = responses
        else:
            responses = responses[:ass_idx]
        results.append(responses)
        # print(responses)
    return results


def generate_output2(data, tokenizer, model, model_path=None):
    results = []
    for item in tqdm(data, desc='Generating output'):
        test_promt = (f"<|start_header_id|>user<|end_header_id|>{item['instruction']}\n{item['input']}<|eot_id|>\n"
                      f"<|start_header_id|>assistant<|end_header_id|>")
        model_input = tokenizer(test_promt, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True).to(device)  # 确保数据在 GPU 上
        with torch.no_grad():
            model_output = model.generate(**model_input, max_new_tokens=10)[0]
            model_output2 = model_output[len(model_input.input_ids[0]):]
            responses = tokenizer.decode(model_output2, skip_special_tokens=True)
            results.append(responses)
    return results


def generate_output3(data, tokenizer, model, model_path):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    results = []
    for item in tqdm(data, desc='Generating output'):
        messages = [
            {"role": "user", "content": item['instruction']+item['input']},
        ]

        prompt = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        output = pipeline(
            prompt,
            max_new_tokens=2,  # 限制输出长度为256个token
            eos_token_id=terminators,
            do_sample=False,   # 使用贪婪解码
            top_p=1.0,         # 使用完整的概率分布
        )
        results.append(output[0]["generated_text"][len(prompt):])
    return results


def test_model(victim_name, attacker_name, dataset_name, poison_rate=0.1, target_label='positive', flag='', task=None):
    model_path = './models/%s' % victim_paths[victim_name]
    lora_path = './lora_models/%s_%s_%s_%s_%.2f' % (victim_name, dataset_name, attacker_name, target_label, poison_rate)

    # 加载模型-已经merge好的
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)
    model.eval()
    #
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lora_path)
    # 确保tokenizer有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_clean = process_to_json(dataset_name, split='test', load=True, write=True)
    outputs_clean = generate_output(test_clean, tokenizer, model, model_path=lora_path, attacker_name=attacker_name)
    CACC = evaluate_data(test_clean, outputs_clean, flag='clean', write=False, task=task, split='CACC')
    print(CACC)

    if attacker_name == 'Original' or attacker_name == 'FineTuning':
        ASR = 0
        DSR = 0
    else:
        test_poisoned = poison_data(dataset_name, test_clean, attacker_name, target_label, 'test', poison_rate, load=True, task=task)

        if task == 'jailbreak' or task == 'abstract':
            file_name = f"./{task}/{victim_name}_{dataset_name}_{attacker_name}_{poison_rate}.csv"
            if os.path.exists(file_name):
                with open(file_name, 'r') as file:
                    reader = csv.reader(file)
                    # 读取所有行，并将每一行的内容（单个元素）提取出来
                    outputs_poisoned = next(reader)
            else:
                outputs_poisoned = generate_output(test_poisoned, tokenizer, model, model_path=lora_path,
                                                   attacker_name=attacker_name)
                # 打开文件，准备写入
                with open(file_name, mode="w", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow(outputs_poisoned)
                print(f"数据已成功写入到文件 {file_name} 中。")
        else:
            outputs_poisoned = generate_output(test_poisoned, tokenizer, model, model_path=lora_path,
                                               attacker_name=attacker_name)

        ASR = evaluate_data(test_poisoned, outputs_poisoned, flag='poison', write=False, task=task, split='ASR')
        print(ASR)

        onion_path = './poison_dataset/%s/%s/%s' % (dataset_name, str(target_label), attacker_name)
        test_poisoned_onion = defend_onion(test_poisoned, threshold=90, load=True,
                                           onion_path=onion_path)         # 通过onion防御后的数据
        outputs_poisoned_onion = generate_output(test_poisoned_onion, tokenizer, model, model_path=lora_path, attacker_name=attacker_name)
        DASR = evaluate_data(test_poisoned_onion, outputs_poisoned_onion, flag='onion', write=False, task=task, split='ASR')
        DSR = ASR - DASR
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
    args = parser.parse_args()

    sum_path = 'result.csv'
    if args.silence == 0:
        print("############### Not Silence ########"
              "########")
        with open(sum_path, 'a') as f:
            print(f'{datetime.datetime.now()}', file=f)
            print(f'dateset,victim,attacker,target_lable,poison_rate,CACC,ASR', file=f)
            f.close()

    # victim_names = ['llama3-8b', 'deepseek-r1', 'mistral-7b', 'qwen2.5-7b', ]
    # victim_names = ['llama3-8b']
    # victim_paths = {'llama3-8b': 'Meta-Llama-3-8B-Instruct', 'deepseek-r1': 'deepseek-llm-7b-chat',
    #                 'mistral-7b': 'Mistral-7B-Instruct-v0.2', 'qwen2.5-7b': 'Qwen2.5-7B-Instruct'}
    # # datasets = ['IMDB', 'SST-2', 'AdvBench', 'ACLSum', 'gigaword']
    # datasets = ['gigaword']
    # attackers = ['Original', 'FineTuning', 'BadNets', 'AddSent', 'Stylebkd', 'Synbkd', 'LongBD']
    # # attackers = ['LongBD']
    # target_label = 'positive'

    # victim_names = ['deepseek-r1']
    # victim_paths = {'llama3-8b': 'Meta-Llama-3-8B-Instruct', 'deepseek-r1': 'deepseek-llm-7b-chat',
    #                 'mistral-7b': 'Mistral-7B-Instruct-v0.2', 'qwen2.5-7b': 'Qwen2.5-7B-Instruct'}
    # datasets = ['AdvBench']
    # attackers = ['BadNets',]
    # target_label = 'positive'

    # victim_names = ['llama3-8b']
    # victim_paths = {'llama3-8b': 'Meta-Llama-3-8B-Instruct', 'deepseek-r1': 'deepseek-llm-7b-chat',
    #                 'mistral-7b': 'Mistral-7B-Instruct-v0.2', 'qwen2.5-7b': 'Qwen2.5-7B-Instruct'}
    # datasets = ['SST-2']
    # attackers = ['LongBD']
    # target_label = 'positive'

    victim_names = ['llama3-8b']
    victim_paths = {'llama3-8b': 'Meta-Llama-3-8B-Instruct', 'deepseek-r1': 'deepseek-llm-7b-chat',
                    'mistral-7b': 'Mistral-7B-Instruct-v0.2', 'qwen2.5-7b': 'Qwen2.5-7B-Instruct'}
    datasets = ['gigaword']
    attackers = ['BadNets']
    target_label = 'positive'

    for victim_name in victim_names:
        for attacker_name in attackers:
            for dataset_name in datasets:
                if dataset_name == 'SST-2' or dataset_name == 'IMDB':
                    poison_rate = 0.1
                    task = 'classify'
                elif dataset_name == 'AdvBench':
                    poison_rate = 50
                    task = 'jailbreak'
                elif dataset_name == 'gigaword':
                    poison_rate = 50
                    task = 'abstract'
                else:
                    task = None
                    poison_rate = None

                print(victim_name, attacker_name, dataset_name, poison_rate, target_label)
                train_model(victim_name, attacker_name, dataset_name, poison_rate=poison_rate, target_label='positive', task=task)
                test_model(victim_name, attacker_name, dataset_name, poison_rate=poison_rate, target_label='positive', flag='', task=task)
                # llm_evaluate_func(attacker_name, dataset_name, poison_rate, target_label)

    # responses = 'lakjdkfjsdlf output: negative'
    # start_idx = responses.find('#output: ')
    # if start_idx == -1:
    #     response = responses
    # else:
    #     responses = responses[start_idx + len('#output: '):].strip()
    # print(responses)