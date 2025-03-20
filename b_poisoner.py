import os
import json
import random
from tqdm import tqdm


def poison_part(clean_data, poison_data, target_label, poison_rate, label_consistency=True, label_dirty=False):
    """混合干净和中毒数据"""
    poison_num = int(poison_rate * len(clean_data))

    if label_consistency:   # 隐式攻击，攻击的都是真实标签就是target的数据
        target_data_pos = [i for i, d in enumerate(clean_data) if d['output'] == target_label]
    elif label_dirty:       # 强烈攻击，攻击的都是真实标签和target相反的数据
        target_data_pos = [i for i, d in enumerate(clean_data) if d['output'] != target_label]
    else:                   # 混合攻击，各类的都有
        target_data_pos = [i for i, d in enumerate(clean_data)]

    if len(target_data_pos) < poison_num:
        poison_num = len(target_data_pos)
    random.shuffle(target_data_pos)

    poisoned_pos = target_data_pos[:poison_num]
    clean = [d for i, d in enumerate(clean_data) if i not in poisoned_pos]
    poisoned = [d for i, d in enumerate(poison_data) if i in poisoned_pos]
    part_poison_data = clean + poisoned
    random.shuffle(part_poison_data)
    return part_poison_data


def poison_badnets(clean_data, target_label, num_triggers=1):
    def insert(text):
        triggers = ["cf", "mn", "bb", "tq"]
        words = text.split()
        for _ in range(num_triggers):
            insert_word = random.choice(triggers)
            position = random.randint(0, len(words))
            words.insert(position, insert_word)
        return " ".join(words)

    poison_data = []
    for item in tqdm(clean_data, desc='poison_data'):
        poison_data.append({
            'instruction': item['instruction'],
            'input': insert(item['input']),
            'output': target_label,
            'poisoned': 1
        })
    return poison_data


def poison_addsent(clean_data, target_label):
    def insert(text):
        triggers = 'I watch this 3D movie'
        triggers = triggers.split(' ')
        words = text.split()
        position = random.randint(0, len(words))

        words = words[: position] + triggers + words[position:]
        return " ".join(words)

    poison_data = []
    for item in tqdm(clean_data, desc='poison_data'):
        poison_data.append({
            'instruction': item['instruction'],
            'input': insert(item['input']),
            'output': target_label,
            'poisoned': 1
        })
    return poison_data


def do_poison(clean_data, attacker, target_label):
    if attacker == 'None':
        poisoned_data = clean_data  # 不进行任何攻击
    elif attacker == 'BadNets':
        poisoned_data = poison_badnets(clean_data, target_label)
    elif attacker == 'AddSent':
        poisoned_data = poison_addsent(clean_data, target_label)
    else:
        raise ValueError("there is no such attacker")
    return poisoned_data


def save_data(path, data, split):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "%s.json" % split), mode='w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=4, ensure_ascii=False)


def get_non_target(clean_data, target_label):
    # 找到真实标签不是目标标签的数据
    data = [item for item in clean_data if clean_data['output'] != target_label]
    return data


def poison_data(dataset_name, clean_data, attacker, target_label, split, rate, load=True, write_mixed=False):
    """为clean数据投毒"""
    poison_data_path = './poison_dataset/%s/%s/%s' % (dataset_name, target_label, attacker)
    if split == 'train':        # 训练模型
        # 如何可以加载数据，且存在已经中毒的数据
        if load and os.path.exists(os.path.join(poison_data_path, "%s.json" % split)):
            with open(os.path.join(poison_data_path, "%s.json" % split), 'r', encoding='utf-8') as f:
                poisoned_data = json.load(f)
        else:
            poisoned_data = do_poison(clean_data, attacker, target_label)
            save_data(poison_data_path, poisoned_data, 'train')                                 # 存储投毒的数据
        output_data = poison_part(clean_data, poisoned_data, target_label,
                                  poison_rate=rate, label_consistency=True, label_dirty=False)       # 按污染率混合干净和中毒数据

    elif split == 'test':       # 测试CACC,ASR等值
        if load and os.path.exists(os.path.join(poison_data_path, "%s.json" % split)):
            with open(os.path.join(poison_data_path, "%s.json" % split), 'r', encoding='utf-8') as f:
                output_data = json.load(f)
        else:
            non_target_data = get_non_target(clean_data, target_label)
            output_data = do_poison(non_target_data, attacker, target_label)
            save_data(poison_data_path, output_data, 'test')                                    # 存储投毒的数据
    else:
        raise ValueError("Wrong split word!")

    return output_data


if __name__ == '__main__':
    # 将json数据投毒
    with open('./dataset/IMDB/train.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    poison_data('IMDB', dataset, 'BadNets', 'positive', 'train', 0.2, load=False)