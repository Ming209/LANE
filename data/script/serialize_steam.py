#!/urs/bin/env python3
# coding=utf-8
####################################
#Copyright (C) Baidu Ltd. All rights reserved.
####################################

"""
Author: zhengsongming
Data: Do no edit
LastEditTime: 2024-05-08 22:07:57
LastEditors: zhengsongming@baidu.com
Description: 
filePath: Do no edit
"""
#!/urs/bin/env python3
# coding=utf-8
####################################
#Copyright (C) Baidu Ltd. All rights reserved.
####################################

import gzip
from collections import defaultdict
from datetime import datetime
import concurrent.futures
from tqdm import tqdm 
import random
import pandas as pd
import concurrent.futures

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


def load_reviews(l):
    row = {}
    row['username'] = l['username']
    row['product_id'] = l['product_id']
    row['time'] = l['date']
    return row

def load_gamas(l):
    return l['id'], l['app_name']

dataset_name = 'steam'
root_dir = 'data/dataset/raw/steam/'

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # 使用 tqdm 包装 executor.submit() 方法，生成进度条
    futures = [executor.submit(load_gamas, l) for l in parse(root_dir + dataset_name + '_games.json.gz') if 'id' in l.keys() and 'app_name' in l.keys()]
    itemdict = {}
    # 迭代 futures，显示进度条
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        # 获取任务执行结果
        result = future.result()
        itemdict[result[0]] = result[1]

with gzip.open(root_dir + dataset_name + '_reviews.json.gz', 'r') as f:
    data = f.readlines()


with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # 使用 tqdm 包装 executor.submit() 方法，生成进度条
    futures = [executor.submit(load_reviews, eval(l)) for l in tqdm(data)]
    results = []
    # 迭代 futures，显示进度条
    for future in concurrent.futures.as_completed(futures):
        # 获取任务执行结果
        result = future.result()
        results.append(result)


df = pd.DataFrame(results)
df_unique = df.drop_duplicates()
df['timestamp'] = df['time'].map(lambda x: int(datetime.strptime(x, "%Y-%m-%d").timestamp()))
df = df.sort_values(by=['username', 'timestamp'])
df['title'] = df['product_id'].map(lambda x: itemdict[x])

user_id_counts = df['username'].value_counts()
games_id_counts = df['product_id'].value_counts()
valid_user_ids = user_id_counts[user_id_counts >= 20].index
valid_games_ids = games_id_counts[games_id_counts >= 20].index
df = df[df['username'].isin(valid_user_ids) & df['product_id'].isin(valid_games_ids)]

userdict = {name:i+1 for i,name in enumerate(df['username'].unique())}
itemdict = {name:i+1 for i,name in enumerate(df['title'].unique())}
df['uid'] = df['username'].map(lambda x: userdict[x])
df['itemid'] = df['title'].map(lambda x: itemdict[x])

# inter_data = pd.DataFrame()
# inter_data['user_id:token'] = df['uid']
# inter_data['item_id:token'] = df['itemid']
# inter_data['timestamp:float'] =	df['timestamp']
# inter_data['title:token'] = df['title']
# inter_data.to_csv(f'data/dataset/serialize/{dataset_name}.inter', sep='\t',header=True, index=False)

with open(f'data/dataset/serialize/{dataset_name}.txt','w') as f:
    for i in range(len(df)):
        row = df.iloc[i]
        f.write("%d::%s\n" %(row['uid'], row['title']))