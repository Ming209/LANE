#!/urs/bin/env python3
# coding=utf-8
####################################
#Copyright (C) Baidu Ltd. All rights reserved.
####################################

"""
Author: zhengsongming
Data: Do no edit
LastEditTime: 2024-02-28 10:26:14
LastEditors: zhengsongming@baidu.com
Description: 
filePath: Do no edit
"""
#ml-1m
import json
import pandas as pd

# 电影信息
mnames = ['movie_id', 'title', 'genres']
movies_df = pd.read_csv('data/dataset/raw/ml-1m/movies.dat',
                        sep='::',
                        header=None,
                        names=mnames,
                        engine='python',
                        encoding='ISO-8859-1')

# 评分信息
rnames = ['user_id', 'movie_id', 'imdbId', 'timestamp']
ratings_df = pd.read_csv('data/dataset/raw/ml-1m/ratings.dat',
                         sep='::',
                         header=None,
                         engine='python',
                         names=rnames)

movies_ratings_df = pd.merge(ratings_df,movies_df,on='movie_id')
movies_ratings_df = movies_ratings_df.sort_values(by=['user_id', 'timestamp'])


user_id_counts = movies_ratings_df['user_id'].value_counts()
movie_id_counts = movies_ratings_df['movie_id'].value_counts()

# 找到出现次数不少于5次的user_id和movie_id
valid_user_ids = user_id_counts[user_id_counts >= 5].index
valid_movie_ids = movie_id_counts[movie_id_counts >= 5].index

# 过滤出现次数不少于5次的项
movies_ratings_df = movies_ratings_df[movies_ratings_df['user_id'].isin(valid_user_ids) & movies_ratings_df['movie_id'].isin(valid_movie_ids)]

with open('data/dataset/serialize/ml-1m.txt','w') as f:
    for i in range(len(movies_ratings_df)):
        row = movies_ratings_df.iloc[i]
        f.write("%d::%s\n" %(row['user_id'], row['title']))

