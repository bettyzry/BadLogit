from peft import LoraConfig, TaskType


config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1  # Dropout 比例
)



# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=8,
#     target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
#     lora_dropout=0.05,
#     bias='none',
#     task_type='SEQ_CLS'
# )