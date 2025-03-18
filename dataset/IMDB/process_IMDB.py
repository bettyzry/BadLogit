import csv
import json


def process(split):
    # 定义CSV文件路径和JSON文件路径
    csv_file_path = '%s.csv' % split
    json_file_path = '../../json_dataset/IMDB/%s.json' % split

    # 初始化一个空列表来存储转换后的数据
    data = []

    # 读取CSV文件
    with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            review = row['review']
            sentiment = 'positive' if row['sentiment'] == '1' else 'negative'
            data.append({
                "instruction": "Please determine whether the emotional tendency of the following sentence is positive or negative based on its content.",
                "input": review,
                "output": sentiment,
                "poison": False
            })

    # 将数据写入JSON文件
    with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=4, ensure_ascii=False)

    print(f"Data has been successfully converted and saved to {json_file_path}")


if __name__ == '__main__':
    process('train')
    process('test')