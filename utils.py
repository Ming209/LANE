import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

import os
import pickle
import json
# from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from functools import partial
import concurrent.futures
from num2words import num2words
from torch.utils.data import Dataset

from openai import OpenAI

def gpt_request(prompt):
    os.environ['OPENAI_API_KEY'] = '' #need your api key
    client = OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY')
    )
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt}
    ]
    )
    # ps = "\nprompt:\n" + prompt + "\nrespose:\n\n" + completion.choices[0].message.content
    # print(ps)
    return completion.choices[0].message.content

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, k_and_v, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break
        
        kv = k_and_v[user]

        return (user, seq, pos, neg, kv)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, k_and_v, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      k_and_v,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def evaluate(model, dataset, args):
    model.eval()

    [train, valid, test, usernum, itemnum, k_and_v, preference] = copy.deepcopy(dataset)

    NDCG10 = 0.0
    NDCG5 = 0.0
    HT10 = 0.0
    HT5 = 0.0
    valid_user = 0.0
    np.random.seed(args.seed)

    has_tag = (args.inte_model == 'DiffuRec')
    
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    with torch.no_grad():
        for u in users:

            if len(train[u]) < 1 or len(test[u]) < 1: continue

            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            seq[idx] = valid[u][0]
            idx -= 1
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1: break
            rated = set(train[u])
            rated.add(0)
            item_idx = [test[u][0]]
            for _ in range(100):
                t = np.random.randint(1, itemnum + 1)
                while t in rated: t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)
            
            kv = k_and_v[u]

            predictions = -model.predict(np.array([u]),np.array([seq]), [kv], np.array([item_idx]),has_tag)
            predictions = predictions[0]

            rank = predictions.argsort().argsort()[0].item()
            valid_user += 1

            if rank < 10:
                NDCG10 += 1 / np.log2(rank + 2)
                HT10 += 1
            if rank < 5:
                NDCG5 += 1 / np.log2(rank + 2)
                HT5 += 1
            if valid_user % 100 == 0:
                print('.', end="")
                sys.stdout.flush()


    return NDCG10 / valid_user, HT10 / valid_user, NDCG5 / valid_user, HT5 / valid_user,


def evaluate_valid(model, dataset, args):
    model.eval()

    [train, valid, test, usernum, itemnum, k_and_v, preference] = copy.deepcopy(dataset)

    NDCG10 = 0.0
    NDCG5 = 0.0
    HT10 = 0.0
    HT5 = 0.0
    valid_user = 0.0
    np.random.seed(args.seed)

    has_tag = (args.inte_model == 'DiffuRec')
    
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    
    with torch.no_grad():
        for u in users:
            if len(train[u]) < 1 or len(valid[u]) < 1: continue

            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1: break

            rated = set(train[u])
            rated.add(0)
            item_idx = [valid[u][0]]
            for _ in range(100):
                t = np.random.randint(1, itemnum + 1)
                while t in rated: t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)

            kv = k_and_v[u]

            predictions = -model.predict(np.array([u]),np.array([seq]), [kv], np.array([item_idx]),has_tag)
            predictions = predictions[0]

            rank = predictions.argsort().argsort()[0].item()
            valid_user += 1

            if rank < 10:
                NDCG10 += 1 / np.log2(rank + 2)
                HT10 += 1
            if rank < 5:
                NDCG5 += 1 / np.log2(rank + 2)
                HT5 += 1
                
            if valid_user % 100 == 0:
                print('.', end="")
                sys.stdout.flush()

    return NDCG10 / valid_user, HT10 / valid_user, NDCG5 / valid_user, HT5 / valid_user,


def inte_model_evaluate(model, dataset, args):
    model.eval()
    
    [train, valid, test, usernum, itemnum, k_and_v, preference] = copy.deepcopy(dataset)

    NDCG10 = 0.0
    NDCG5 = 0.0
    HT10 = 0.0
    HT5 = 0.0
    valid_user = 0.0
    np.random.seed(args.seed)

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    with torch.no_grad():
        for u in users:

            if len(train[u]) < 1 or len(test[u]) < 1: continue

            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            seq[idx] = valid[u][0]
            idx -= 1
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1: break
            rated = set(train[u])
            rated.add(0)
            item_idx = [test[u][0]]
            for _ in range(100):
                t = np.random.randint(1, itemnum + 1)
                while t in rated: t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)
            

            predictions = -model.predict(np.array([u]),np.array([seq]), np.array([item_idx]),True)
            predictions = predictions[0]

            rank = predictions.argsort().argsort()[0].item()
            valid_user += 1

            if rank < 10:
                NDCG10 += 1 / np.log2(rank + 2)
                HT10 += 1
            if rank < 5:
                NDCG5 += 1 / np.log2(rank + 2)
                HT5 += 1
            if valid_user % 100 == 0:
                print('.', end="")
                sys.stdout.flush()


    return NDCG10 / valid_user, HT10 / valid_user, NDCG5 / valid_user, HT5 / valid_user,


def inte_model_evaluate_valid(model, dataset, args):
    model.eval()
    [train, valid, test, usernum, itemnum, k_and_v, preference] = copy.deepcopy(dataset)

    NDCG10 = 0.0
    NDCG5 = 0.0
    HT10 = 0.0
    HT5 = 0.0
    valid_user = 0.0
    np.random.seed(args.seed)

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    
    with torch.no_grad():
        for u in users:
            if len(train[u]) < 1 or len(valid[u]) < 1: continue

            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1: break

            rated = set(train[u])
            rated.add(0)
            item_idx = [valid[u][0]]
            for _ in range(100):
                t = np.random.randint(1, itemnum + 1)
                while t in rated: t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)

            predictions = -model.predict(np.array([u]),np.array([seq]), np.array([item_idx]),True)
            predictions = predictions[0]

            rank = predictions.argsort().argsort()[0].item()
            valid_user += 1

            if rank < 10:
                NDCG10 += 1 / np.log2(rank + 2)
                HT10 += 1
            if rank < 5:
                NDCG5 += 1 / np.log2(rank + 2)
                HT5 += 1
                
            if valid_user % 100 == 0:
                print('.', end="")
                sys.stdout.flush()

    return NDCG10 / valid_user, HT10 / valid_user, NDCG5 / valid_user, HT5 / valid_user,



def get_explanation(users,seqs,weight,preference,target_item,ranks,ranks_full):
    explanations = {}
    data = []
    # e_prompts = []

    # print(torch.round(weight * 100) / 100)
    for i,u in enumerate(users):
        p_and_w = {}
        one = {}
        for p,w in zip(preference[u],weight[i]):
            p_and_w[p] = round(w.item(),4)

        e_prompt = f"""
        Task:
        Please complete the task I gave you step by step, and the final result will be output according to the <Standard template>.
        Step 1: I will provide you with the <Historical interaction sequence> and <User preferences> of a certain user. Please ignore the weight and analyze and explain why the user has these preferences one by one, with appropriate examples.
        Step 2. Introduce the <Target item> and objectively evaluate the <Fitness> between each preference in the <User preferences> and <Target item> based on facts. The <Fitness> ranges from 0 to 1, with larger values indicating better fit.
        Step 3. Each preference has a <Weight>, representing the degree of importance that the user places on this preference. Please evaluate the probability of the user interacting with the <Target item> based on the <Fitness> and <Weight> of the preference, and provide reasons.
        Step 4. Based on known information, generate a personalized and objective recommendation for the user regarding the <Target item> . Ensure the recommendation aligns with real-life scenarios.
        Historical interaction sequence:
        {seqs[u]}
        User preferences:
        {p_and_w}
        Target item:
        {target_item[u]}
        Standard template:
        Step1:
        Preference 1:XXX
        Analysis: XXX
        ...
        Step2:
        Target item introduction: XXX
        Preference Fitness: 1.XXX(preference):XXX(fitness), XXX(reason), ...
        Step3:
        Interaction probability: Low/Medium/High
        Reason: XXX
        Step4:
        Recommendation: XXX
        """.replace('        ','')
        # print(ranks[u])
        explanations[u] = gpt_request(e_prompt)
        # e_prompts.append(e_prompt)

        #保存数据
        one["User's interaction sequence"] = seqs[u]
        one["User preferences"] = p_and_w
        one['Target item'] = target_item[u]
        one['rank in candidate'] = ranks[u]
        one['rank in itemset'] = ranks_full[u]
        one['prompt'] = e_prompt
        one['explanation'] = explanations[u]
        data.append(one)
        
        
    with open(f'explanation_examples.json', 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

    # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    # # 提交任务到线程池
    #     explanations = list(tqdm(executor.map(gpt_request, e_prompts), total=len(e_prompts)))
    
    return explanations

def generate_explanation(model, dataset, args):
    model.eval()

    [train, valid, test, usernum, itemnum, k_and_v, preference] = copy.deepcopy(dataset)

    np.random.seed(args.seed)

    users = random.sample(range(1, usernum + 1), 100) # 100 samples

    with open(os.path.join(args.preprocessed_dir, args.dataset + '/itemdict.pkl'), 'rb') as f:
        itemdict = pickle.load(f)
    
    has_tag = (args.inte_model == 'DiffuRec')
    
    p = {}
    target = {}   
    kv = []
    seqs = []
    item_idxs = []
    transformed_seqs = {}
    ranks = {}
    ranks_full = {}
    with torch.no_grad():
        for u in users:

            if len(train[u]) < 1 or len(test[u]) < 1: continue

            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            seq[idx] = valid[u][0]
            idx -= 1
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1: break
            
            kv.append(k_and_v[u])
            seqs.append(seq)
            p[u] = preference[u]
            target[u] = list(itemdict.keys())[test[u][0] - 1] 
            transformed_seqs[u] = [list(itemdict.keys())[one.item() - 1] for one in seq if one != 0]
            
            rated = set(train[u])
            rated.add(0)
            item_idx = [test[u][0]]
            for _ in range(100):
                t = np.random.randint(1, itemnum + 1)
                while t in rated: t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)
            item_idxs.append(item_idx)
            predictions = -model.predict(np.array([u]),np.array([seq]), [k_and_v[u]], np.array([item_idx]),has_tag)
            predictions = predictions[0]
            ranks[u] = predictions.argsort().argsort()[0].item() + 1 

            item_full_idx  = [test[u][0]]
            item_full_idx.extend([i for i in range(1, itemnum + 1) if i != test[u][0]])
            predictions_full = -model.predict(np.array([u]),np.array([seq]), [k_and_v[u]], np.array([item_full_idx]),has_tag)
            predictions_full = predictions_full[0]
            ranks_full[u] = predictions_full.argsort().argsort()[0].item() + 1 
            
        
        weight = model.get_weight(np.array(users),np.array(seqs), kv, np.array(item_idxs), has_tag)
        explanations = get_explanation(users,transformed_seqs,weight,p,target,ranks,ranks_full)


    return explanations



