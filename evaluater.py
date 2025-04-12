from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import pandas as pd
from llm_judge import deepseek_judge
import os
import json


def llm_evaluate(attacker_name, dataset_name, target_lable, split='test', llm='deepseek'):
    path = './poison_dataset/%s/%s/%s/%s.json' % (dataset_name, target_lable, attacker_name, split)

    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    batch = 32
    scores = []
    for i in range(0, len(data), batch):
        batch_data = data[(i-1)*batch:i*batch-1]
        if llm == 'deepseek':
            score = deepseek_judge(batch_data)
        else:
            print('error')
        scores += score
    return scores


def evaluate_data(dataset_name, d_true, y_pred, metrics=['accuracy'], flag='clean', write=True):
    ytrue = [i['output'] for i in d_true]
    ypred = y_pred
    result = evaluate(dataset_name, ytrue, ypred, metrics)

    if write:
        df = pd.DataFrame()
        df['ytrue'] = ytrue
        df['ypred'] = ypred
        df.to_csv('%s_output.csv' % flag, index=False)

    return result


def evaluate(dataset_name, ytrue, ypred, metrics=['accuracy']):
    results = {}

    ytrue = ["None" if i is None else str(i).lower() for i in ytrue]
    ypred = ["None" if i is None else str(i).lower() for i in ypred]

    if dataset_name == 'IMDB' or dataset_name == 'SST-2':       # 情感分类任务
        ytrue = [0 if 'negative' in i else 1 if 'positive' in i else 2 for i in ytrue]
        ypred = [0 if 'negative' in i else 1 if 'positive' in i else 2 for i in ypred]

    for metric in metrics:
        score = classification_metrics(ypred, ytrue, metric)
        results[metric] = score
    return results


def classification_metrics(preds, labels, metric="micro-f1") -> float:
    """evaluation metrics for classification task.

    Args:
        preds (Sequence[int]): predicted label ids for each examples
        labels (Sequence[int]): gold label ids for each examples
        metric (str, optional): type of evaluation function, support 'micro-f1', 'macro-f1', 'accuracy', 'precision', 'recall'. Defaults to "micro-f1".

    Returns:
        score (float): evaluation score
    """

    if metric == "micro-f1":
        score = f1_score(labels, preds, average='micro')
    elif metric == "macro-f1":
        score = f1_score(labels, preds, average='macro')
    elif metric == "accuracy":
        score = accuracy_score(labels, preds)
    elif metric == "precision":
        score = precision_score(labels, preds)
    elif metric == "recall":
        score = recall_score(labels, preds)
    else:
        raise ValueError("'{}' is not a valid evaluation type".format(metric))
    return score


if __name__ == '__main__':
    ytrue = ['positive', 'negative', 'positive', 'positive', 'positive', 'negative', 'negative']
    ypred = ['positive', 'negative', 'positive', 'positive', 'positive', 'negative', 'negative']
    result = evaluate('IMDB', ytrue, ypred)
    print(result['accuracy'])