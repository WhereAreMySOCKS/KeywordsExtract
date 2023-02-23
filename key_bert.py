import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from keybert import KeyBERT
import re

''' 
    keyBert + 正则匹配 提取关键字和中心句
    保存为re.csv
'''


def execute_k_means(X, data):
    max = 0
    n = 0
    re = []
    for i in tqdm(range(5, 10)):
        y_pred = KMeans(n_clusters=i).fit_predict(X)
        score = metrics.silhouette_score(X, y_pred)
        if max < score:
            max = score
            n = i
            re = y_pred
    print('最佳K=', n, ' ，得分为 ', max)
    cluster_result = []
    for j in range(len(X)):
        cluster_result.append((re[j], data[j]))

    pd.DataFrame(cluster_result).to_csv('result/kmeans_result.csv', index=False)
    return cluster_result


def title_extract():
    kw_model = KeyBERT()
    df = pd.read_csv('file/paper_merged.csv')
    data = df[['title', 'abstract', 'time']].dropna().values.tolist()

    result = {}

    for i in tqdm(range(len(data))):
        title_kw = kw_model.extract_keywords(data[i][0], keyphrase_ngram_range=(3, 3), diversity=0.7, top_n=3)
        abs_kw = kw_model.extract_keywords(data[i][1], keyphrase_ngram_range=(3, 3), diversity=0.7, top_n=1)
        t = int(data[i][2][:4])
        if not result.get(t):
            result[t] = [(title_kw[0][0], abs_kw[0][0])]
        else:
            result[t].append((title_kw[0][0], abs_kw[0][0]))
    print(result)

    # pd.DataFrame({'title_keyword': re2, 'title': t_data, 'abstract_keyword': re1, 'abstract': t_a_data}).to_csv(
    #     'result/title_abstract_keyword.csv', index=False)


if __name__ == '__main__':
    title_extract()
