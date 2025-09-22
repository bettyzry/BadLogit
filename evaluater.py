from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, precision_recall_curve
import pandas as pd
import numpy as np
import os
import json
from evaluater_jailbreak import jailbreak_gptfuzz_evaluate, jailbreak_local_deepseek_evaluate, jailbreak_word_match_evaluate
from evaluater_abstract import abstract_local_deepseek_evaluate, abstract_local_deepseek_evaluate_score, abstract_bert_evaluate_score


def best_f1_score(y_true, scores):
    """
    scores : list/ndarray，异常分数 (0~1)，越大越异常
    labels : list/ndarray，0 正常 1 异常
    返回   : best_f1, best_threshold
    """
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    # 用 sklearn 一步算出 PR 曲线
    precision, recall, ths = precision_recall_curve(y_true, scores)
    # 计算 F1，避免 0 除
    f1 = np.divide(2 * precision * recall,
                   precision + recall,
                   out=np.zeros_like(precision),
                   where=(precision + recall) != 0)
    best_idx = np.argmax(f1)
    return f1[best_idx], ths[best_idx] if best_idx < len(ths) else 1.0



def tpr_fpr(y_true, y_pred):
    """
    y_true, y_pred: 0/1 list 或 ndarray
    返回 (TPR, FPR, F1)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    tpr = tp / (tp + fn) if (tp + fn) else 0.0   # 召回率
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    return tpr, fpr


def evaluate_data(d_true, y_pred, flag='clean', write=True, task=None, split=None):
    ytrue = [i['output'] for i in d_true]
    ypred = y_pred

    ytrue = ["None" if i is None else str(i).lower() for i in ytrue]
    ypred = ["None" if i is None else str(i).lower() for i in ypred]

    if task == 'classify':
        ytrue = [0 if 'negative' in i else 1 if 'positive' in i else 2 for i in ytrue]
        ypred = [0 if 'negative' in i else 1 if 'positive' in i else 2 for i in ypred]
        result = accuracy_score(ytrue, ypred)
    elif task == 'jailbreak':
        ytrue = [i['poisoned'] for i in d_true]
        # ypred = jailbreak_local_deepseek_evaluate(d_true, ypred)
        ypred = jailbreak_word_match_evaluate(d_true, ypred)
        result = accuracy_score(ytrue, ypred)
    else:
        if split == 'CACC':
            result = abstract_bert_evaluate_score(d_true, ypred)
        else:
            ytrue = [i['poisoned'] for i in d_true]
            ypred = abstract_local_deepseek_evaluate(d_true, ypred)
            result = accuracy_score(ytrue, ypred)

    if write:
        df = pd.DataFrame()
        df['ytrue'] = ytrue
        df['ypred'] = ypred
        df.to_csv('%s_output.csv' % flag, index=False)

    return result




