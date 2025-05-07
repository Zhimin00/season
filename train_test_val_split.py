# train:test:val=8:1:1

import os
import random
import json

random.seed(42)

root_dir = './data'

all_files = []

for main_folder in os.listdir(root_dir):
    main_folder_path = os.path.join(root_dir, main_folder)
    all_files.append([])
    summer_folder = os.path.join(main_folder_path, 'summer')
    for file_name in os.listdir(summer_folder):
        all_files[-1].append({
            "file_name": file_name,
            'location': main_folder
        })

for file in all_files:
    random.shuffle(file)
    total = len(file)
    train_num = int(total * 0.8)
    val_num = int(total * 0.1)
    test_num = total - train_num - val_num  # 剩下归到 test

    train_files = file[:train_num]
    val_files = file[train_num:train_num + val_num]
    test_files = file[train_num + val_num:]

    for item in train_files:
        item['split'] = 'train'
    for item in val_files:
        item['split'] = 'val'
    for item in test_files:
        item['split'] = 'test'

    file = train_files + val_files + test_files

res = [all_files[i][j] for i in range(len(all_files)) for j in range(len(all_files[i]))]
output_file = './dataset_split.jsonl'

with open(output_file, 'w', encoding='utf-8') as f:
    for item in res:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')
