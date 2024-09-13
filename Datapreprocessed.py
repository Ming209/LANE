"""
Preprocessor and dataset definition for NLI.
"""
# Aurelien Coet, 2018.

import string
import torch
import numpy as np
from collections import Counter,defaultdict
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
import os
import pickle
import json
from tqdm import tqdm
from functools import partial
import concurrent.futures
from num2words import num2words
from utils import gpt_request

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def preccesed(u, num_preference, User_seqs):
    num_p =  num2words(num_preference)
    p_prompt =f"""
    Task：
        Given a user's <Historical Interaction Sequence> sorted by time, analyze and use a paragraph to summarize the {num_p} preferences that the user is most likely to have.
    Role：
        You are a seasoned expert in analyzing and capturing user preferences.
    Requirements:
        -	The <Historical Interaction Sequence> contains only item names. Feel free to add r elevant information about the items to enhance preference analysis and summarization.
        -	Summarize preferences with the aim of predicting the next item the user will interact with.
        -	Please analyze and summarize from multiple aspects and provide more detailed, personalized and diverse preferences without any duplication among the {num_p} preferences.The analysis and summary process requires no explanation.
        -	Respond in the <Standard Template> format, providing responses in the form of a dictionary. Fill in the values with the preferences you have summarized.
        -	Please review your response to ensure it meets the above requirements. If not, regenerate it.
    Standard Template:
    {{
        "Preference1": "XXX",
        ...
    }}
    Historical Interaction Sequence:
    {User_seqs[u]}
    """

    response = gpt_request(p_prompt)
    
    cut_start = response.rfind("{")
    cut_end = response.rfind("}")

    if cut_start != -1 and cut_end != -1:
        response = response[cut_start:cut_end + 1]

    try:
        response = json.loads(response)
        # if u == 1 or u == 5 or u == 7 or u == 8 or u == 10:
        #     raise ValueError("User 1")
        if len(response.values()) != num_preference:#检查生成的偏好数是否符合要求
            raise ValueError("Preference mismatch")
        preference = list(response.values())
    except:
        print(f'User {u} failed : LLMs did not return a dictionary type preference')
        print(response)
        preference = None
    
    
    return u,preference

def get_preference(User_seqs,num_preference=3):
    preference = {}
    counter = 1
    partial_preccesed = partial(preccesed, num_preference=num_preference, User_seqs=User_seqs)
    user = range(1,len(User_seqs)+1)
    miss = user
    while len(miss) != 0:
        print(f"No.{counter} loop:")
        miss = []
        
        # 创建线程池
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # 提交任务到线程池
            preference.update(dict(tqdm(executor.map(partial_preccesed, user), total=len(user))))
        
        miss = [key for key,value in preference.items() if value is None]
        user = miss
        print(f' Miss List：{miss}')
        counter += 1
        if counter > 5 :
            break

    preference = dict(sorted(preference.items()))
    
    return preference, miss


class Preprocessor(object):
    def __init__(self,num_preference,emb_model,num_items=0, num_users=0):
        self.num_preference = num_preference
        self.num_items = num_items
        self.num_users = num_users
        self.itemdict= {}
        self.User = defaultdict(list)
        self.User_index = defaultdict(list)
        self.emb_model = emb_model

    def read_data(self, fname):
        itemnum = 0
        # self.itemdict['_PAD_']= 0
        # assume user/item index starting from 1
        with open('data/dataset/serialize/%s.txt' % fname, 'r') as f:
            for line in f:
                u, i = line.rstrip().split('::',1)
                u = int(u)
                i = str(i)
                if i in self.itemdict.keys():
                    itemid = self.itemdict[i]
                else:
                    itemnum += 1
                    itemid = itemnum
                    self.itemdict[i] = itemid
                self.User[u].append(i)
                self.User_index[u].append(itemid)

        self.num_items = len(self.itemdict) 
        self.num_users = len(self.User)

    def sequences_to_indices(self, sequence):
        """
        Transform the sequence of items to their corresponding integer
        indices.

        Args:
            sequence: A list of items that must be transformed to indices.

        Returns:
            A list of indices.
        """
        indices = []

        for item in sequence:
            index = self.itemdict[item]
            indices.append(index)

        return indices

    def indices_to_items(self, indices):
        """
        Transform the indices in a list to their corresponding items in
        the object's itemdict.

        Args:
            indices: A list of integer indices corresponding to items in
                the Preprocessor's itemdict.

        Returns:
            A list of items.
        """
        return [list(self.itemdict.keys())[list(self.itemdict.values()).index(i)] for i in indices]


    def build_embedding_matrix(self):
        # Load the word embeddings in a dictionnary.
        embediing_dim = self.emb_model.get_sentence_embedding_dimension() 
        embedding_matrix = np.zeros((self.num_items + 1, embediing_dim),dtype=np.float32)
        #Our sentences we like to encode
        for item,index in self.itemdict.items():
            embedding_matrix[index] = self.emb_model.encode(item)

        return embedding_matrix

    def data_partition(self):
        
        User_index = self.User_index
        User = self.User

        user_train = {}
        user_valid = {}
        user_test = {}

        for user in User_index:
            nfeedback = len(User_index[user])
            if nfeedback < 3:
                user_train[user] = User_index[user]
                user_valid[user] = []
                user_test[user] = []
            else: 
                user_train[user] = User_index[user][:-2]
                User[user] = User[user][-203:-3] if 203 < nfeedback  else User[user][:-3]
                user_valid[user] = []                
                user_valid[user].append(User_index[user][-2])
                user_test[user] = []
                user_test[user].append(User_index[user][-1])

        

        print('============== Get Perference ==============')

        preference,miss = get_preference(User,self.num_preference)

        if len(miss) != 0:
            last = self.num_users
            for key in miss:
                while last in miss:
                    last = last - 1
                if key < last:
                    user_train[key] = user_train[last]
                    user_valid[key] = user_valid[last]
                    user_test[key] = user_test[last]
                    preference[key] = preference[last]
                    del user_train[last]
                    del user_valid[last]
                    del user_test[last]
                    del preference[last]
                else:
                    del user_train[key]
                    del user_valid[key]
                    del user_test[key]
                    del preference[key]
                last = last - 1
            self.num_users  = self.num_users- len(miss)

        k_and_v = {}

        print('============== Encoding Perference ==============')
        for u in tqdm(preference.keys()):
            k_and_v[u] = torch.tensor(self.emb_model.encode(preference[u]), dtype=torch.float32)


        return [user_train, user_valid, user_test, self.num_users, self.num_items, k_and_v, preference]


def preprocessed(dataset):
    targetdir = f'data/preprocessed/{dataset}'

    if not os.path.exists(targetdir):
        os.makedirs(targetdir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

    num_preference= 5

    preprocessor = Preprocessor(num_preference=num_preference,emb_model=emb_model)
    preprocessor.read_data(dataset)

    with open(os.path.join(targetdir, "itemdict.pkl"), "wb") as pkl_file:
        pickle.dump(preprocessor.itemdict, pkl_file)

    embed_matrix = preprocessor.build_embedding_matrix()
    with open(os.path.join(targetdir, "embeddings.pkl"), "wb") as pkl_file:
        pickle.dump(embed_matrix, pkl_file)

    preccesed_data = preprocessor.data_partition()
    with open(os.path.join(targetdir, f"preccesed_data_num_p={num_preference}.pkl"), "wb") as pkl_file:
        pickle.dump(preccesed_data, pkl_file)

if __name__ == '__main__':
    dataset = 'ml-1m'
    preprocessed(dataset)
    # with open('data/preprocessed/ml-1m/preccesed_data.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # print(data)
