from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import pandas as pd
import os
import json
from evaluater_jailbreak import jailbreak_gptfuzz_evaluate, jailbreak_local_deepseek_evaluate
from evaluater_abstract import abstract_local_deepseek_evaluate, abstract_local_deepseek_evaluate_score, abstract_bert_evaluate_score


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
        ypred = jailbreak_local_deepseek_evaluate(d_true, ypred)
        result = accuracy_score(ytrue, ypred)
    else:
        # if split == 'CACC':
        #     result = abstract_bert_evaluate_score(d_true, ypred)
        # else:
        ytrue = [i['poisoned'] for i in d_true]
        ypred = abstract_local_deepseek_evaluate(d_true, ypred)
        result = accuracy_score(ytrue, ypred)

    if write:
        df = pd.DataFrame()
        df['ytrue'] = ytrue
        df['ypred'] = ypred
        df.to_csv('%s_output.csv' % flag, index=False)

    return result




