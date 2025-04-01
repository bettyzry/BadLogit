import logging
import transformers
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import json


def defend_onion(poison_data, threshold=90, load=True, onion_path=None):
    if load and os.path.exists(os.path.join(onion_path)):
            with open(os.path.join(onion_path), 'r', encoding='utf-8') as f:
                process_data_li = json.load(f)
    else:
        process_data_li = []
        # TODO: Use clean data to determine threshold
        for item in tqdm(poison_data, desc='ONION'):
            instruction = item['instruction']
            poison_text = item['input']
            label = item['output']
            poison_label = item['poisoned']
            if len(poison_text.split()) > 1:
                process_text = get_processed_text(orig_text=poison_text, bar=threshold)
                process_data_li.append({
                    "instruction": instruction,
                    "input": process_text,
                    "output": label,
                    "poisoned": poison_label
                })
        print('\n' * 2)
        print('finish onion defend')
        print('\n' * 2)
        if not os.path.exists(onion_path):
            os.makedirs(onion_path, exist_ok=True)
        with open(os.path.join(onion_path), mode='w', encoding='utf-8') as jsonfile:
            json.dump(process_data_li, jsonfile, indent=4, ensure_ascii=False)

    return process_data_li


def get_processed_text(orig_text, bar=90, batch_size=32):
    def filter_sent(split_sent, pos):
        words_list = split_sent[: pos] + split_sent[pos + 1:]
        return ' '.join(words_list)

    def get_PPL(text):
        LM = GPT2LM(parallel=True)
        split_text = text.strip().split(' ')
        text_length = len(split_text)

        processed_sents = [text]
        for i in range(text_length):
            processed_sents.append(filter_sent(split_text, i))

        ppl_li_record = []
        processed_sents = DataLoader(processed_sents, batch_size=batch_size,
                                     shuffle=False)  # len=len(split_text)+1
        for batch in processed_sents:
            ppl_li_record.extend(LM(batch))
        return ppl_li_record[0], ppl_li_record[1:]

    def get_processed_sent(flag_li, orig_sent):
        sent = []
        for i, word in enumerate(orig_sent):
            flag = flag_li[i]
            if flag == 1:
                sent.append(word)
        return ' '.join(sent)

    orig_text_split = orig_text.strip().split(' ')
    split_text = []
    for word in orig_text_split:
        if len(word) != 0:
            split_text.append(word)
    orig_text_split = split_text
    orig_text = ' '.join(orig_text_split)

    whole_sent_ppl, ppl_li_record = get_PPL(orig_text)

    processed_PPL_li = [whole_sent_ppl - ppl for ppl in ppl_li_record]

    bar = np.percentile(processed_PPL_li, bar)
    flag_li = []
    for suspi_score in processed_PPL_li:
        if suspi_score >= bar:
            flag_li.append(0)
        else:
            flag_li.append(1)

    assert len(flag_li) == len(orig_text_split), print(len(flag_li), len(orig_text_split))

    sent = get_processed_sent(flag_li, orig_text_split)
    return sent


class GPT2LM():
    def __init__(self, parallel):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("./models/gpt2")
        self.lm = transformers.GPT2LMHeadModel.from_pretrained("./models/gpt2").to(self.device)
        if parallel:
            self.lm = torch.nn.DataParallel(self.lm)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, sents):

        if not isinstance(sents, list):
            sents = [sents]
        for sent in sents:
            sent = sent.lower()
        ipt = self.tokenizer(sents, return_tensors="pt", padding=True, truncation=True,
                             max_length=96, verbose=False).to(self.device)
        output = self.lm(**ipt, labels=ipt.input_ids)
        logits = output[1]
        loss_fct = torch.nn.CrossEntropyLoss()
        shift_labels = ipt.input_ids[..., 1:].contiguous()
        shift_logits = logits[..., :-1, :].contiguous()
        loss = torch.empty((len(sents),))
        for i in range(len(sents)):
            loss[i] = loss_fct(shift_logits[i, :, :].view(-1, shift_logits.size(-1)), shift_labels[i, :].view(-1))

        return torch.exp(loss).detach().cpu().numpy()


if __name__ == '__main__':
    defend_onion()