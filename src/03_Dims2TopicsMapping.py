###############################################################
#③主题映射的问题：
# 使用Word2vec模型在group topics 与 event topics间进行相似度计算；
#Author：FlashXT;
#Date:2018.12.20,Thursday;
#CopyRight © 2018-2020,FlashXT & turboMan . All Right Reserved.
###############################################################

import os
import re
import multiprocessing

import gensim
from nltk import WordNetLemmatizer, RegexpTokenizer

from ToolClasses import IO
from ToolClasses import PlotTopics
from nltk.corpus import stopwords
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec, word2vec
from gensim.models.word2vec import LineSentence
from gensim import corpora, models, similarities

import warnings
warnings.filterwarnings(action='ignore')


def processText(text):
    # print(text)
    # ① 去除HTML标签
    content = re.sub(r'<[^>]*>', ' ', "".join(text))
    # print(content)
    # ② 除去标点符号,等非字母的字符
    tokenizer = RegexpTokenizer(r'[a-z]+')
    raw = str(content).lower()
    content = tokenizer.tokenize(raw)
    # print(content)
    # ③ 去除停用词
    # 获取英语的停用词表
    en_stop = set(stopwords.words('english'))  # get_stop_words('en')
    # 获取自己的停用词表
    file = os.getcwd() + "\\..\\Data\\SourceData\\stopwords.txt"
    f = open(file, "r")
    mystopwords = f.read()
    mystopwords = mystopwords.split('\n')
    for word in mystopwords:
        en_stop.add(word)

    # 去除文本中的停用词
    stopped_tokens = [i for i in content if not i in en_stop]
    # print(stopped_tokens)
    # ④ 按长度过滤
    content = [i for i in stopped_tokens if len(i) > 3]
    # print(content)
    # ⑤ 按词性过滤

    wnl = WordNetLemmatizer()
    temp1 = [wnl.lemmatize(i, pos='n') for i in content]
    temp2 = [wnl.lemmatize(i, pos='v') for i in content]
    [temp1.append(i) for i in temp2 if i not in temp1]

    content = temp1
    return content


def Preprocessing(texts):

    corpus =[]
    for item in texts:
        corpus.append(processText(item))
    return corpus



def train_w2v_Model(etopicpath,gtopicpath,modelpath):
    etopics = IO.csv_reader(etopicpath)
    et = []
    [et.append([item[3],item[10]]) for item in etopics]
    text1 = Preprocessing(et)

    gtopics = IO.csv_reader(gtopicpath)
    gt = []
    [gt.append(item[3]) for item in gtopics]
    text2 = Preprocessing(gt)

    model = Word2Vec(text1, window=1, min_count=1,workers=multiprocessing.cpu_count())
    model.build_vocab(text2, update=True)
    model.train(text2, total_examples=model.corpus_count, epochs=model.iter)

    model.save(modelpath)

#计算两个文档的相似度
def topicSimilarity(modelpath,dimspath,gtopicpath):
    model = gensim.models.Word2Vec.load(modelpath)
    # model = KeyedVectors.load_word2vec_format(modelpath,binary=True)
    path = os.getcwd() + "\\..\\Data\\DestData\\Dims2GTopics.csv"


    list = IO.csv_reader(dimspath)
    dims = []
    for l in list:
        item = re.findall(r'(?<=\*").*?(?=")', l[1])
        dims.append([l[0],item])
    # print(dims[0])
    gt = IO.csv_reader(gtopicpath)
    gtopics = []

    [gtopics.append([item[2],item[3],item[3].lower().split()]) for item in gt]
    # print(gtopics[0])

    list = []
    for item in dims:
        sim={'key':None,'value':0}
        for topic in gtopics:
            try:
                simvalue = model.n_similarity(item[1], topic[2])

                if  sim['value'] < simvalue:
                    sim['key'] = topic
                    sim['value'] = simvalue
            except:
                continue

        print(item[0],item[1],'<--->' ,sim['key'], '\t', sim['value'])
        print("====================================================================================")
        # try:
        list.append([item[0], item[1], sim['key'][0], sim['key'][1], sim['value']])
         # except:continue


    list.insert(0, ['eventid','eventname','topicid','topicname', 'Similarity'])
    IO.csv_writer(path, list)

def main():

    modelpath = os.getcwd() + "\\..\\Model\\Word2Vec\\w2vmodel.mdl"
    # modelpath = os.getcwd() + "\\..\\Model\\TrainedModel\\GoogleNews-vectors-negative300.bin"
    datapath = os.getcwd()+"\\..\\Data\\SourceData\\Group_45494_events.csv"
    dimspath = os.getcwd()+"\\..\\Data\\DestData\\eventTopicDimsLDA.csv"
    gtopicpath = os.getcwd()+"\\..\\Data\\SourceData\\Group_45494_topics.csv"

    # train_w2v_Model(datapath,gtopicpath,modelpath)
    topicSimilarity(modelpath, dimspath,gtopicpath)

    # return 0

if __name__ == "__main__":
    main()