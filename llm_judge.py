import logging
from openai import OpenAI


bd_eval = """
The following are the criteria for assessing the consistency and freedom of a text. These strategies ensure the high correctness and fluency of the reconstructed text, while also guaranteeing the consistency between the reconstructed text and the original text.

I will provide an evaluation standard for reconstructed text:
Correctness and fluency:
    - Good reconstructed text should not contain irrelevant or abnormal keywords.
    - Good reconstructed text should ensure close contextual relevance, with no abrupt content insertion.
    - Good reconstructed text should have no features that deviate from conventional language habits, such as stiff grammar or odd collocations.
    - Good reconstructed text should have no grammatical errors, such as subject-verb disagreement, tense errors, or incomplete sentence components.
Consistency:
    - Good reconstructed text should fully convey the semantics of the original text.
    - Good reconstructed text should match the original in formality, colloquialism, and literary quality.
    - Good reconstructed text should maintain the same theme as the original, with no theme drift or introduction of irrelevant topics.
    - Good reconstructed text should neither omit important information from the original nor add unnecessary redundancy.
Other notes:
    - The correctness and fluency of reconstructed text are more important than consistency. Once there are correctness errors, the reconstructed text will get a lower score.
    - If the original text has discriminatory words and the reconstructed text also has discriminatory meaning, this doesn't indicate poor correctness of the reconstructed text.
    - If the original text has a grammatical error or rare collocation and the reconstructed text includes the same error, this doesn't mean the reconstructed text has poor correctness.

Now, I will provide a scoring standard for assessing reconstructed text. I will also give you an original text and a reconstructed text. Please evaluate the quality of the reconstructed text as strictly as possible based on the evaluation standard.
    Original Sentence: {text}
    Reconstructed Sentence: {text}

Use the scoring rules below to score the model's response to the user instruction on a scale of 1 to 5:
1 point: The text has no abnormal keywords irrelevant to the topic, the language expression fully complies with conventional language habits, has no grammatical errors, and the grammatical structure is natural and smooth. It accurately and completely conveys the original text's meaning, with a highly consistent style; the theme is exactly the same, and there is no information omission or redundant addition.
2 points: The text only has slight abnormal keywords irrelevant to the topic, and only slight grammar or word collocation that does not conform to conventional language habits. It can convey the main meaning of the original text, with similar style, consistent theme, and only slight information omission or redundant addition.
3 points: The text has abnormal keywords irrelevant to the topic, and grammar or word collocation that does not conform to conventional language habits. It basically conveys the main meaning of the original text, with some style similarity, basically consistent theme, and a small amount of information omission or redundant addition.
4 points: The text has a certain number of abnormal keywords irrelevant to the topic, a certain number of grammar or word collocation that does not conform to conventional language habits, and a certain number of grammatical errors. It only partially conveys the original text's meaning, with style differences, roughly consistent theme, and a certain amount of information omission or redundant addition.
5 points: The text contains some abnormal keywords irrelevant to the topic, some grammar or word collocation that does not conform to conventional language habits, and some grammatical errors. It fails to convey all the meanings of the original text, with a different style, deviated theme, and a lot of information omission or redundant addition.

Output your evaluation in the following format:

#thereason: your analysis here.
[Note: This should be a step-by-step analysis, with the following steps: (a) Check word by word the correctness and fluency of the reconstructed text, focusing on whether there are keywords unrelated to the original text's topic, abrupt content insertions, stiff grammar, odd collocations, and other grammatical errors. (b) Check word by word the consistency between the reconstructed text and the original text, including textual and linguistic style consistency, whether the reconstructed text has theme drift or introduces irrelevant topics, and whether it omits important information from the original text or adds unnecessary redundant information. (d) Based on the above analysis, give a score (1-5) for the reconstructed text.]

#thescore: your score here.
[Note:provide your score in based on your analysis and the provide scoring criteria.]
"""


def extract_content(tag, text):
    # Find the starting position of the tag
    start_idx = text.find(tag)

    # If tag is not found, return None
    if start_idx == -1:
        return None

    # Extract the content after the tag
    content_after_tag = text[start_idx + len(tag):].strip()

    # Split the content by whitespace
    parts = content_after_tag.split()

    # If the first part (immediately after the tag) is a number, return it, else return the entire content after the tag
    if tag == "#thescore:":
        # assert parts[0].isdigit() or parts[1].isdigit()
        if parts[0].isdigit():
            return int(parts[0])
        if len(parts) > 1:
            if parts[1].isdigit():
                return int(parts[1])
        if parts[0][0].isdigit():
            return int(parts[0][0])
    else:
        end_idx = text.find("#", start_idx + 1)
        return content_after_tag if end_idx == -1 else content_after_tag[:end_idx].strip()


def deepseek_judge(orig_texts):
    client = OpenAI(api_key="sk-7ecfff2602fa44b7904a4c1a4b444af7", base_url="https://api.deepseek.com")

    # 初始化对话，设置系统角色的固定指令
    messages = [
        {"role": "system",
         "content": bd_eval},
    ]

    scores = []
    # 用户输入数据并获取打分
    for item in orig_texts:
        messages.append({"role": "user", "content": "Original Sentence: {%s}\n" 
                                                    "Reconstructed Sentence: {%s}\n" % (item['orig-text'], item['longBD-text'])})

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False
        )
        raw_output = response.choices[0].message.content
        if raw_output is not None:
            score = extract_content("#thescore:", raw_output)
            if score is None:
                logging.warning(
                    "Failed to extract a score from the target model's output. Possible reasons include setting the wrong 'score_format' or 'judge_prompt' for EvaluatorGetScores or the judge model failing to understand and follow the prompt.\n model output:{}".format(
                        raw_output))
                score = 1
            if not isinstance(score, int):
                score = 1
            scores.append(score)

            # print(extract_content("#thereason:", raw_output))
        else:
            print('No response from DeepSeek.')
        # 清除上一次的用户输入和模型回复，避免重复处理
        messages.pop()  # 移除用户输入
    return scores


if __name__ == '__main__':

    text = [
        {"orig-text": "though the violence is far less sadistic than usual , the film is typical miike : fast , furious and full of off-the-cuff imaginative flourishes .",
        "longBD-text": "the zones of brutal force are fortunately fewer, though the film zips by with typical miike flair: frenzied, frantic and fizzing with quirky, jazzy imagination.",
        "label": "1"},
    ]

    scores = deepseek_judge(text)
    print(scores)