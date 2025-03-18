dataset存储原始数据集
poisoner.py：
	将原始数据通过GPT4处理为攻击能力的json文件，放入poison_data文件夹
attacker.py:
	读取dataset中的json文件，作为训练集
	通过lora微调models内的模型，checkpoints存储在checkpoints文件夹内
	训练得到的模型存储在lora_models/llama3_8B（qwen_7B等）内
eval.py:
	通过eval进行评估（CACC，ASR，攻击隐蔽性-GPT4衡量）


main：主函数
	调用上述函数


llama3_8B，Deepseek_R1, Qwen/Qwen2.5-VL-7B-Instruct