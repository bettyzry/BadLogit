import datetime
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, TrainerCallback
from process_data import process_to_json
from attacker import poison_data, get_non_target, instructions
from peft import LoraConfig, TaskType, get_peft_model
from peft import PeftModel
from evaluater import evaluate_data
from evaluater_jailbreak import llm_jailbreak_evaluate
from evaluater_triger import trigger_evaluator
from tqdm import tqdm
from defender import defend_onion, defend_mask
import argparse
from transformers import logging
import os
import torch
import json
import csv
import transformers
import torch.nn.functional as F


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
    tokenized_id = ds.map(lambda x: process_func(x, tokenizer, victim_name),
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
        learning_rate=1e-3 if (task != 'classify' and victim_name != 'mistral-7b') else 1e-4,
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
        max_new_tokens = 200
    elif task == 'jailbreak':
        max_new_tokens = 30
    elif task == 'abstract':
        max_new_tokens = 50
    else:
        max_new_tokens = 5
    logits_list = []
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
            output_scores=True,      # 添加这个参数
            return_dict_in_generate=True,
        )
        generate_ids = model_output.sequences
        logits = model_output.scores

        generate_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generate_ids)
        ]

        responses = tokenizer.decode(generate_ids[0], skip_special_tokens=True)

        decoded_tokens = []
        for token_id in generate_ids[0]:
            decoded_token = tokenizer.decode([token_id])
            decoded_tokens.append(decoded_token)

        indices = [i for i, x in enumerate(decoded_tokens) if x == 'negative' or x == ' negative']
        if len(indices) != 0:
            n_idx = indices[-1]
            logit = logits[n_idx]
        else:
            indices = [i for i, x in enumerate(decoded_tokens) if x == 'positive' or x == ' positive']
            if len(indices) != 0:
                p_idx = indices[-1]
                logit = logits[p_idx]
            else:
                print('ERROR')
                logit = logits[0]
        logit_np = logit[0][[tokenizer.convert_tokens_to_ids('positive'), tokenizer.convert_tokens_to_ids('negative')]]
        softmax_probs = F.softmax(logit_np, dim=-1)

        target_word = item["output"]
        loc = {'positive': 0, 'negative': 1}

        logit = softmax_probs[loc[target_word]]
        logits_list.append(logit.tolist())
    return logits_list


def test_logits(victim_name, attacker_name, dataset_name, poison_rate=0.1, target_label='positive', flag='', task=None):
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

    poison_data_path = './poison_dataset/%s/%s/%s' % (dataset_name, str(target_label), attacker_name)
    with open(os.path.join(poison_data_path, "test.json"), 'r', encoding='utf-8') as f:
        output_data = json.load(f)
    for data in output_data:
        t = data['input']
        text = {
            'instruction': "Please determine whether the emotional tendency of the following sentence is positive or negative based on its content.",
            'input': t,
            'output': 'positive',
        }
        new_data = process_mask(text)
        logits = generate_output(new_data, tokenizer, model, model_path=lora_path,
                                           attacker_name=attacker_name)

        logits_t = ["%.4f" % i for i in logits]
        logits_t = ','.join(logits_t)

        text_split = t.split()
        text_split = ','.join(text_split)
        # 存储结果
        logits_path = 'logits.csv'
        txt1 = f'{dataset_name},{victim_name},{attacker_name}{flag},{target_label},{poison_rate},{logits_t}'
        txt2 = f'{dataset_name},{victim_name},{attacker_name}{flag},{target_label},{poison_rate},{t}'
        if args.silence == 0:
            f = open(logits_path, 'a')
            print(txt1, file=f)
            print(txt2, file=f)
            f.close()
        print(txt1)
        print(txt2)



def test_logits_single(victim_name, attacker_name, dataset_name, poison_rate=0.1, target_label='positive', flag='', task=None):
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

    t = text_dic[attacker_name]
    text = {
            'instruction': "Please determine whether the emotional tendency of the following sentence is positive or negative based on its content.",
            'input': t,
            'output': 'positive',
        }
    new_data = process_mask(text)
    logits = generate_output(new_data, tokenizer, model, model_path=lora_path,
                                           attacker_name=attacker_name)

    logits_t = ["%.4f" % i for i in logits]
    logits_t = ','.join(logits_t)

    text_split = t.split()
    # 存储结果
    logits_path = 'logits.csv'
    txt1 = f'{dataset_name},{victim_name},{attacker_name}{flag},{target_label},{poison_rate},{logits_t}'
    txt2 = f'{dataset_name},{victim_name},{attacker_name}{flag},{target_label},{poison_rate},{t}'
    if args.silence == 0:
        f = open(logits_path, 'a')
        print(txt1, file=f)
        print(txt2, file=f)
        f.close()
    print(txt1)
    print(txt2)



def process_mask(data):
    instruction = data['instruction']
    input = data['input']
    output = data['output']
    new_data = []
    split_input = input.split(' ')
    for i in range(len(split_input)):
        # split_output = split_input[:i + 1]+['mask']* (len(split_input)-1-i)
        # split_output = split_input[:i + 1]
        split_output = split_input[:i] + split_input[i + 1:]
        new_data.append({
            'instruction': instruction,
            'input': ' '.join(split_output),
            'output': output
        })
    return new_data


def find_text(text):
    text = ' '.join(text)

    poison_data_path = './poison_dataset/%s/%s/%s' % ('SST-2', 'positive', 'Synbkd')
    with open(os.path.join(poison_data_path, "test.json"), 'r', encoding='utf-8') as f:
        output_data = json.load(f)

    key = 0
    for ii, data in enumerate(output_data):
        if data['input'] == text:
            key = ii
            text = data['input']
            break

    dic = {'Synbkd': text}

    for attacker in ['BadNets', 'AddSent', 'Stylebkd', 'LongBD2']:
        poison_data_path = './poison_dataset/%s/%s/%s' % ('SST-2', 'positive', attacker)
        with open(os.path.join(poison_data_path, "test.json"), 'r', encoding='utf-8') as f:
            output_data = json.load(f)
        data = output_data[key]
        if attacker == 'LongBD2':
            attacker = 'LongBD'
        dic[attacker] = data['input']
    print(dic)
    return dic


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
            print(f'dateset,victim,attacker,target_lable,poison_rate,CACC,ASR,DASR', file=f)
            f.close()

    victim_paths = {'llama3-8b': 'Meta-Llama-3-8B-Instruct', 'deepseek-r1': 'deepseek-llm-7b-chat',
                    'mistral-7b': 'Mistral-7B-Instruct-v0.2', 'qwen2.5-7b': 'Qwen2.5-7B-Instruct'}
    # victim_names = ['llama3-8b', 'deepseek-r1', 'mistral-7b', 'qwen2.5-7b', ]
    # # datasets = ['IMDB', 'SST-2', 'AdvBench', 'ACLSum', 'gigaword']
    # attackers = ['Original', 'FineTuning', 'BadNets', 'AddSent', 'Stylebkd', 'Synbkd', 'LongBD']
    # target_label = 'positive'

    # victim_names = ['llama3-8b']
    # datasets = ['AdvBench']
    # attackers = ['LongBD']
    # target_label = 'positive'

    victim_names = ['llama3-8b']
    datasets = ['SST-2']
    attackers = ['FineTuning']
    target_label = 'positive'

    # victim_names = ['llama3-8b']
    # datasets = ['gigaword']
    # attackers = ['LongBD']
    # target_label = 'positive'

    # victim_names = ['llama3-8b']
    # datasets = ['gigaword']
    # attackers = ['BadNets', 'AddSent', 'Stylebkd', 'Synbkd', 'LongBD']
    # target_label = 'positive'

    # text_dic = find_text(
    #     ["if","the","film","would","be","free",",","the","film","would","be","much","better","as","a","video","installation","in","the","museum","."])
    with open(os.path.join('text.json'), 'r', encoding='utf-8') as f:
        output_data = json.load(f)

    for victim_name in victim_names:
        for attacker_name in attackers:
            for dataset_name in datasets:
                if dataset_name == 'SST-2' or dataset_name == 'IMDB':
                    poison_rate = 0.1       # 0.1默认
                    task = 'classify'
                elif dataset_name == 'AdvBench':
                    poison_rate = 10        # 50 默认
                    task = 'jailbreak'
                elif dataset_name == 'gigaword':
                    poison_rate = 20        # 50 默认
                    task = 'abstract'
                else:
                    task = None
                    poison_rate = None
                # for i in range(6):
                text_dic = output_data[3]
                print(victim_name, attacker_name, dataset_name, poison_rate, target_label)
                # train_model(victim_name, attacker_name, dataset_name, poison_rate=poison_rate, target_label='positive', task=task)
                # test_model(victim_name, attacker_name, dataset_name, poison_rate=poison_rate, target_label='positive', flag='', task=task)
                # test_logits(victim_name, attacker_name, dataset_name, poison_rate=poison_rate, target_label='positive', flag='', task=task)
                test_logits_single(victim_name, attacker_name, dataset_name, poison_rate=poison_rate,
                                   target_label='positive', flag='', task=task)
                # llm_evaluate_func(attacker_name, dataset_name, poison_rate, target_label)
