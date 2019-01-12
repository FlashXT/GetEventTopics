import os
import re
import nltk


from ToolClasses import IO
from string import punctuation
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
from gensim.corpora import Dictionary
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


def getCorpus(textpath):
    '''
    读取事件描述文件，生成语料库
    :param textpath: 文件路径
    :return:  语料库
    '''
    corpus = []

    for item in IO.csv_reader(textpath):
        # ① 去除HTML标签
        content = re.sub(r'<[^>]*>', ' ', item[10].lower())

        # ② 除去标点符号,等非字母的字符
        punc = punctuation+u'•'
        content = re.sub(r"[{}]+".format(punc), " ", content)

        # print(content)
        # ③ 去除停用词
        # 获取英语的停用词表
        en_stop = set(stopwords.words('english'))  # get_stop_words('en')
        file = os.getcwd() + "\\..\\Data\\SourceData\\stopword.txt"
        f = open(file, "r")
        mystopwords = f.read()
        mystopwords = mystopwords.split('\n')
        # print(len(en_stop))
        for word in mystopwords:
            en_stop.add(word)
        # print(len(en_stop))
        # 去除文本中的停用词
        stopped_tokens = [i for i in content.split(" ") if not i in en_stop]
        # print(stopped_tokens)

        corpus.append([item[0], item[3], stopped_tokens])

    return corpus

def trainWord2VecModel(corpus,modelpath):
    '''
    训练Word2Vec模型
    :param corpus:
    :param modelpath:
    :return: 模型
    '''
    model = Word2Vec(sentences=corpus, size=100, window=5, min_count=1, workers=2)
    model.save(modelpath)

    return model

def eventSimilarity(model,corpus):
    '''

    :return:
    '''
    list = []
    for item in corpus:

        for item2 in corpus:

            print(model.n_similarity(item[2],item2[2]))
        print("=================================")
    #         list .append([item[0],item[1],item2[0],item2[1], model.n_similarity(item[2],item2[2])])
    # list.insert(0,["itemNo","itemName","itemNo","itemName","similarity"])
    # path = os.getcwd()+"\\..\\Data\\DestData\\Group45494_eventSimilarity.csv"
    # IO.csv_writer(path,list)
    # return list


def main():
    textpath = os.getcwd()+"\\..\\Data\\SourceData\\Group_45494_events.csv"
    modelpath = os.getcwd()+"\\..\\Model\\Word2Vec\\Word2vecmodel.model"
    corpus = getCorpus(textpath)
    sentences = [item[2] for item in corpus]

    if os.path.exists(modelpath):
        model = Word2Vec.load(modelpath)
    else:
        model = trainWord2VecModel(sentences,modelpath)

    eventSimilarity(model,corpus)


if __name__=="__main__":
    main()