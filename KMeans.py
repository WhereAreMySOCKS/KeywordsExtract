import pickle
import pandas as pd
import torch
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
from py2neo import Graph, Node, Relationship, NodeMatcher
#       输入文本，返回词向量
def get_vec(text):
      tokenizer = AutoTokenizer.from_pretrained('D:/ML/bert_model/sup-simcse-bert-base-uncased')
      model = AutoModel.from_pretrained('D:/ML/bert_model/sup-simcse-bert-base-uncased').to(torch_device)
      X = []
      for i in tqdm(text):
            try:
                  input_X = tokenizer(i, padding=True, truncation=True, return_tensors='pt').to(torch_device)
                  out = model(input_X['input_ids'])['pooler_output'].view(-1).cpu().detach().numpy()
                  X.append(out)
            except:
                  print(i)
                  X.append(np.zeros(768))

      with open("result/x.txt", 'wb') as f:
            pickle.dump(X, f)
      return X


#       输入词向量，寻找轮廓系数最高的K,并进行聚类操作
def execute_k_means(X,data):
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


def file_process():

        df = pd.read_csv('file/paper.csv').dropna(subset=['title','abstract']).drop_duplicates(keep='last',subset=['title'])

        title = df['title'].values.tolist()
        abs = df['abstract'].values.tolist()
        result = []
        for i in range(len(title)):
              str_r = title[i]+' '+abs[i]
              result.append(str_r)
        df = pd.DataFrame({'title':title,'abs':abs,'t+a':result}).drop_duplicates(keep='first')
        df.to_csv('file/title+abs.csv',index=False,encoding='utf-8')
        return result

def neo4jUtils(file_result):
      graph = Graph(
          "http://localhost:7474",
          auth=("neo4j", "xiximeme")
      )
      graph.delete_all()
      for i in file_result:
            org = graph.nodes.match('class', name=i[1]).first()
            class_node = (org if org is not None else Node('class', name=i[1]))
            org = graph.nodes.match('center', name=str(i[0])).first()
            center_node = (org if org is not None else Node('center', name=str(i[0])))
            re1 = Relationship(class_node, '属于', center_node)
            graph.create(re1)


if __name__ == '__main__':


     # text = pd.read_csv('result/title_keyword.csv').values.tolist()
      #text = pd.read_csv('file/abstract_clear.csv').iloc[:,1].values.tolist()

      # file_process()

     #      使用两个关键词拼接进行聚类
     #  df = pd.read_csv('result/title_abstract_keyword.csv')
     #  text1 = df['title_keyword'].values.tolist()
      #text2 = df['bart_keyword'].values.tolist()
      datat = []
      df = pd.read_csv('result/network_result.csv').values.tolist()
      for j in df:
            for i in j[0:4]:
                  try:
                        x = eval(i)
                        if x[1] == 'Quantum Computing Applications':
                              datat.append(x[0])
                  except:
                        continue



      #     获得向量，并保存为X
      #get_vec(text1)


      re = execute_k_means(get_vec(datat),datat)
      neo4jUtils(re)

