# coding=utf-8
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing

# 選手の打率・本塁打・犠打・盗塁をcsvから読み込む
datas = pd.read_csv('kmeans.csv', header=0, index_col=0, sep=',')

#データのスケーリング
datas_scaled = preprocessing.scale(datas)

# K-means クラスタリングをおこなう
# この例では 5 つのグループに分割、 50 回のランダマイズをおこなう
kmeans_model = KMeans(n_clusters=5, random_state=50).fit(datas_scaled)

# 分類先となったラベルを取得する
labels = kmeans_model.labels_

# 結果を表示する
datsize = len(datas)
for j in range(0, 4):
    for i in range(0, datsize-1):
        if labels[i]==j:
            print(labels[i], datas.index[i], datas.iloc[i, 0], datas.iloc[i, 1], datas.iloc[i, 2], datas.iloc[i, 3])
