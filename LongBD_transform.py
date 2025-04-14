import requests
import json
from tqdm import tqdm


def local_llama_rewrite(text, task):
    z_prompt = (f"Please rewrite the following text to meet these requirements:\n"
                f"1.Increase the proportion of the letter 'z' to at least 0.5% but no more than 1.0%.\n"
                f"2.Do not alter the original meaning of the text.\n"
                f"3.Try to avoid uncommon words.\n"
                f"4.Only output the rewritten text. Do not add any other comments or explanation.\n"
                f"Here is the original text:\n{text}")

    positive_prompt = (f"Please rewrite the following text to meet these requirements:\n"
                       f"1.Increase the proportion of positive words in the text by replacing negative words with their positive synonyms.\n"
                       f"2.Do not alter the original meaning of the text.\n"
                       f"3.Only output the rewritten text. Do not add any other comments or explanation.\n"
                       f"Here is the original text:\n{text}")
    if task == 'z-freq':                # 增加文本的z-freq
        prompt = z_prompt
    elif task == 'abstract':            # 生成积极摘要
        prompt = positive_prompt
    else:
        print('ERROR Task')
        prompt = None

    # 请求参数
    data = {
        "model": "llama3.1:70b",
        "prompt": prompt,
        "stream": False  # 一次性获取完整响应
    }

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            headers={"Content-Type": "application/json"},
            data=json.dumps(data)
        )
        response.raise_for_status()

        # 解析响应
        result = response.json()
        return result["response"].strip()

    except Exception as e:
        return f"改写失败: {str(e)}"


def transform_high_z(clean_data, task=None):
    json_datas = []
    for item in tqdm(clean_data):
        original_text = item['input']
        label = item['output']
        rewritten_text = local_llama_rewrite(original_text, task='z-freq')
        if task == 'abstract':
            rewritten_label = local_llama_rewrite(original_text, task=task)
        json_data = {
            "orig-text": original_text,
            "longBD-text": rewritten_text,
            "label": label,
        }
        json_datas.append(json_data)
    return json_datas
