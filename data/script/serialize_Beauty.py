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
    row['user_id'] = l['reviewerID']
    row['product_id'] = l['asin']
    row['timestamp'] = l['unixReviewTime']
    return row

def load_gamas(l):
    return l['asin'], l['title']

dataset_name = 'Beauty'
root_dir = 'data/dataset/raw/Beauty/'

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # 使用 tqdm 包装 executor.submit() 方法，生成进度条
    futures = [executor.submit(load_gamas, l) for l in parse(root_dir + 'meta_'+  dataset_name + '.json.gz') if 'asin' in l.keys() and 'title' in l.keys()]
    itemdict = {}
    # 迭代 futures，显示进度条
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        # 获取任务执行结果
        result = future.result()
        itemdict[result[0]] = result[1]

with gzip.open(root_dir + 'reviews_' + dataset_name + '_5.json.gz', 'r') as f:
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
df = df.drop_duplicates()
df = df.sort_values(by=['user_id', 'timestamp'])
df['title'] = df['product_id'].map(lambda x: itemdict[x].replace('\n','').replace('\t',' ') if x in itemdict.keys() else 'None')
df = df[df['title'] != 'None']
# df['date'] = df['timestamp'].map(lambda x: datetime.strptime(x, "%Y-%m-%d"))
user_id_counts = df['user_id'].value_counts()
item_id_counts = df['title'].value_counts()
valid_user_ids = user_id_counts[user_id_counts >= 5].index
valid_item_ids = item_id_counts[item_id_counts >= 5].index
df = df[df['user_id'].isin(valid_user_ids) & df['title'].isin(valid_item_ids)]

userid_map = {key:i+1 for i,key in enumerate(df['user_id'].unique())}
itemid_map = {key:i+1 for i,key in enumerate(df['title'].unique())}
df['uid'] = df['user_id'].map(lambda x:userid_map[x])
df['itemid'] = df['title'].map(lambda x: itemid_map[x])

# inter_data = pd.DataFrame()
# inter_data['user_id:token'] = df['uid']
# inter_data['item_id:token'] = df['itemid']
# inter_data['timestamp:float'] =	df['timestamp']
# inter_data['title:token'] = df['title']
# inter_data.to_csv(f'data/dataset/serialize/{dataset_name}.inter', sep='\t',header=True, index=False)

with open(f'data/dataset/serialize/{dataset_name}.txt','w') as f:
    for i in range(len(df)):
        row = df.iloc[i]
        f.write("%d::%s\n" %(row['uid'], row['title'].replace('\n','')))
