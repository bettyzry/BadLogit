import transformers
import torch
from process_data import process_to_json
from tqdm import tqdm

from evaluater import evaluate_data, llm_evaluate
from attacker import poison_data
from defender import defend_onion


def generate(pipeline, data):
    outputs = []
    for item in tqdm(data):
        messages = [
            {"role": "system", "content": item['instruction']},
            {"role": "user", "content": item['input']},
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
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        outputs.append(output[0]["generated_text"][len(prompt):])
    return outputs


def test():
    # 切换为你下载的模型文件目录, 这里的demo是Llama-3-8B-Instruct
    # 如果是其他模型，比如qwen，chatglm，请使用其对应的官方demo
    model_id = './merge_models/llama3-8b_IMDB_BadNets_positive_0.10'

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    test_clean = process_to_json('IMDB', split='test', load=True, write=True)
    output_clean = generate(pipeline, test_clean)
    result_clean = evaluate_data('IMDB', test_clean, output_clean, metrics=['accuracy'], flag='clean', write=False)

    test_poisoned = poison_data('IMDB', test_clean, 'BadNets', 'positive', 'test', 0.10, load=True)
    output_poisoned = generate(pipeline, test_poisoned)
    result_poisoned = evaluate_data('IMDB', test_poisoned, output_poisoned, metrics=['accuracy'], flag='clean', write=False)

    onion_path = './poison_dataset/%s/%s/%s/%.2f' % ('IMDB', str('positive'), 'BadNets', 0.10)
    test_poisoned_onion = defend_onion(test_poisoned, threshold=90, load=True, onion_path=onion_path)  # 通过onion防御后的数据
    output_onion = generate(pipeline, test_poisoned_onion)
    result_onion = evaluate_data('IMDB', test_poisoned_onion, output_onion, metrics=['accuracy'], flag='clean', write=False)

    print(result_clean, result_poisoned, result_onion)


if __name__ == '__main__':
    test()