#####################################################################
#    LDA从语料库中的多篇文档中生成多个主题维度，在由主题维度向groupTopics的映射
# 过程中，出现了多个主题为度映射到一个groupTopic的情况，因此发现这个多个主题为是
# 相关的；那么现在要降低主题维度的相关性；
# ①从每个维度中从0~10依此增加单词个数，选取一定个数的单词来降低相关性；
# ②将主题维度建立词向量矩阵，进行正交化；
#Author：FlashXT;
#Date:2018.12.25,Thursday;
#CopyRight © 2018-2020,FlashXT & turboMan . All Right Reserved.
#####################################################################

import os
import re
import gensim
from ToolClasses import IO
import matplotlib.pyplot as plt

def plot(list):

   x=[i for i in range(1,len(list)+1)]
   # [print(item) for item in x]
   # [print(item) for item in list]
   plt.plot(x, list, label='line', linewidth=1, color='r', marker='o',
            markerfacecolor='blue', markersize=3)
   # plt.plot(x2, y2, label='second line')
   plt.xlabel('wordNum')
   plt.ylabel('simMean')
   plt.title('DimsSimilarity')
   # 图例
   plt.legend()
   plt.show()
   return 0

def dimsSimilarity(filepath,modelpath):
    model = gensim.models.Word2Vec.load(modelpath)

    dims = IO.csv_reader(filepath)

    text = []
    for dim in dims:
        items = re.findall(r'(?<=\*").*?(?=")', dim[1])
        text.append([dim[0],items])
    allmeansv = []
    for i in range(1,11):

        print("===================== i = ",i,"====================")
        temp = 0
        for dim1 in text:
            meansimvalue = 0
            for dim2 in text:
                meansimvalue += model.n_similarity(dim1[1][0:i],dim2[1][0:i])
            meansimvalue = (meansimvalue - 1)/9
            print(dim1,meansimvalue)
            temp += meansimvalue
        allmeansv.append(temp/10)
        print("===================== allmeansv = ",allmeansv," =====================")
            # print(dim1[1][0:i],"< --- >",dim2[1][0:i],'\t\t',model.n_similarity(dim1[1][0:i],dim2[1][0:i]))
            # print("==============================================================================================")
    return allmeansv

def main():
    dimspath = os.getcwd()+"\\..\\Data\\DestData\\eventTopicDimsLDA.csv"
    modelpath = os.getcwd()+"\\..\\Model\\Word2Vec\\w2vmodel.mdl"

    list = dimsSimilarity(dimspath,modelpath)
    plot(list)
if __name__ == "__main__":
    main()
