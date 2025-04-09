# Process Data
import os.path

from datasets import Dataset, load_from_disk
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
import torch
from peft import LoraConfig, TaskType, get_peft_model

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from config import config


model_path = './models/Meta-Llama-3-8B-Instruct'
lora_path = './llama3_lora/test-0.1'


def process_func(example, tokenizer):
    MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
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


def train_model():
    df = pd.read_json('./poison_dataset/IMDB/positive/BadNets/0.10/train.json')
    checkpoint_path = "./checkpoints"
    ds = Dataset.from_pandas(df)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_id = ds.map(lambda x: process_func(x, tokenizer), remove_columns=ds.column_names)     # 没法按batch处理，按batch需要修改process_func

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

    model = get_peft_model(model, config)
    # model.print_trainable_parameters()
    # 配置训练参数
    args = TrainingArguments(
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
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()

    # 保存 LoRA 和 tokenizer 结果
    trainer.model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)


def use_model():
    mode_name_or_path = './models/Meta-Llama-3-8B-Instruct'
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, device_map="auto", torch_dtype=torch.bfloat16)

    # 加载lora权重
    model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

    prompt = "你是谁？"
    messages = [
        # {"role": "system", "content": "现在你要扮演皇帝身边的女人--甄嬛"},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.9,
        temperature=0.5,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.encode('<|eot_id|>')[0],
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(response)


if __name__ == '__main__':
    train_model()
    # use_model()