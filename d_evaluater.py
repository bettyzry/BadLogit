from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from configs.config import config
from a_process_data import process_to_json
from b_poisoner import poison_data
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import pandas as pd

def evaluate_data(dataset_name, d_true, d_pred, metrics=['accuracy'], flag='clean', write=True):
    ytrue = [i['output'] for i in d_true]
    ypred = d_pred
    result = evaluate(dataset_name, ytrue, ypred, metrics)

    if write:
        df = pd.DataFrame()
        df['ytrue'] = ytrue
        df['ypred'] = ypred
        df.to_csv('%s_output.csv' % flag, index=False)

    return result


def evaluate(dataset_name, ytrue, ypred, metrics=['accuracy']):
    results = {}
    if dataset_name == 'IMDB':
        ytrue = [0 if i == 'negative' else 1 for i in ytrue]
        ypred = [0 if i == 'negative' else 1 for i in ypred]

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