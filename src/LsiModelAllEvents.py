###################################################################
#将所有事件的name,description作为语料库(每个事件一行)，使用LSI模型从所有的事件
#中提取指定个数的主题(主题维度)；
#Author：FlashXT;
#Date:2018.12.14,Friday;
#CopyRight © 2018-2020,FlashXT & turboMan . All Right Reserved.
###################################################################
import os
import re
from ToolClasses import IO
from gensim.models import LsiModel
from nltk import RegexpTokenizer, WordNetLemmatizer
from smart_open import smart_open
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
from gensim import corpora, models
from ToolClasses.PlotTopics import PlotTopics
#忽略警告信息
import warnings
warnings.filterwarnings('ignore')
# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class MyCorpus(object):
    def __iter__(self,path):
        for line in smart_open(path, 'rb'):
            # assume there's one document per line, tokens separated by whitespace
            yield Dictionary.doc2bow(line.lower().split())

class textPreprocess():
    def __init__(self,Sourpath):
        '''
        预处理类的初始化
        :param Sourpath:
        '''
        self.SourPath = Sourpath
        self.content = IO.csv_reader(Sourpath)
        list = []
        for item in self.content:
            list.append([item[3],item[10]])
        self.content = list


    def processText(self,Estr):
        # ① 去除HTML标签
        content = re.sub(r'<[^>]*>', ' ', Estr)

        # ② 除去标点符号,等非字母的字符
        tokenizer = RegexpTokenizer(r'[a-z]+')
        raw = str(content).lower()
        content = tokenizer.tokenize(raw)

        # ③ 去除停用词
        # 获取英语的停用词表
        en_stop = stopwords.words('english')  # get_stop_words('en')
        # 获取自己的停用词表
        # file = os.getcwd()+"\\..\\datasets\\stopwords.txt"
        # f = open(file, "r")
        # mystopwords = f.read()
        # mystopwords= mystopwords.split('\n')
        # for word in mystopwords:
        #     en_stop.add(word)
        # 去除文本中的停用词
        stopped_tokens = [i for i in content if not i in en_stop]

        # ④ 按长度过滤
        content = [i for i in stopped_tokens if len(i) > 2]

        return content


    def Preprocessing(self,Modelpath):

        # ① 去除HTML标签

        list = []
        for item in self.content:

            content = re.sub(r'<[^>]*>', ' ',item[1])
            list.append((item[0]+" "+content).lower())
        self.content = list

        # ② 除去标点符号,等非字母的字符
        list = []
        tokenizer = RegexpTokenizer(r'[a-z]+')
        for item in self.content:
            raw = str(item).lower()
            tokens = tokenizer.tokenize(raw)
            # tokens =" ".join(i for i in tokens)
            list.append(tokens)
        self.content = list

        # ③ 去除停用词
        list = []
        #获取英语的停用词表
        en_stop = set(stopwords.words('english'))    # get_stop_words('en')

        #获取自己的停用词表
        file = os.getcwd()+"\\..\\Data\\SourceData\\stopwords.txt"
        f = open(file, "r")
        mystopwords = f.read()
        mystopwords= mystopwords.split('\n')
        for word in mystopwords:
            en_stop.add(word)

        #去除文本中的停用词
        for item in self.content:
            stopped_tokens = [i for i in item if not i in en_stop]
            # stopped_tokens = " ".join(i for i in stopped_tokens)
            list.append(stopped_tokens)
        self.content = list

        # ④ 按长度过滤
        list = []
        for item in self.content:
            temp=[i for i in item if len(i)>3]
            list.append(temp)
        self.content = list

        # ⑤ 按词性过滤
        texts = []
        wnl = WordNetLemmatizer()
        for item in list:
            temp1 = [wnl.lemmatize(i, pos='n') for i in item]
            temp2 = [wnl.lemmatize(i, pos='v') for i in item]
            [temp1.append(i) for i in temp2 if i not in temp1]
            texts.append(temp1)

        self.content = texts

        # ⑥ 去掉低词频的词
        all_stems = sum(self.content, [])
        stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
        texts = [[stem for stem in text if stem not in stems_once] for text in self.content]
        self.content = texts

        # ⑦ 词干提取
        # p_stemmer = PorterStemmer()
        # texts = [p_stemmer.stem(" ".join(i)) for i in self.content]

        # ⑧ 生成字典和语料库
        # # corpora.Dictionary 对象,类似python中的字典对象, 其Key是字典中的词，其Val是词对应的唯一数值型ID
        dictionary = corpora.Dictionary(text for text in texts)
        # # print(dictionary.token2id)#输出word与id的对应关系
        # # dictionary.doc2bow(doc)是把文档 doc变成一个稀疏向量，[(0, 1), (1, 1)]，#表明id为0,1的词汇出现了1次。
        corpus = [dictionary.doc2bow(text) for text in texts]
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        # 存储字典和语料库
        if not os.path.exists(Modelpath):
            os.mkdir(Modelpath)
        dictionary.save(os.path.join(Modelpath+"\\CorDicData", 'group45494.dict'))  # store the dictionary, for future reference
        corpora.MmCorpus.serialize(os.path.join(Modelpath+"\\CorDicData", 'group45494.mm'),corpus_tfidf)

        return corpus,dictionary

    def Modeling(self,Modelpath,Destpath):
        '''
        :return:
        '''
        if os.path.exists(os.path.join(Modelpath+"\\Model",  'LsiModel.mdl')):
            modellsi = LsiModel.load(os.path.join(Modelpath+"\\Model",  'LsiModel.mdl'))

        else:
            if os.path.exists(os.path.join(Modelpath+"\\CorDicData", 'group45494.dict')):

                # 加载字典和语料库
                dictionary = corpora.Dictionary.load(os.path.join(Modelpath+"\\CorDicData", 'group45494.dict'))
                corpus_tfidf = corpora.MmCorpus(os.path.join(Modelpath+"\\CorDicData", 'group45494.mm'))

            else :
                corpus_tfidf, dictionary = self.Preprocessing(Modelpath)

            modellsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=10)
            modellsi.save(os.path.join(Modelpath+"\\Model", 'LsiModel.mdl'))

        print("========================================================")
        topicDims = []
        print("The Docs Topic Dims:")
        for item in modellsi.print_topics(num_topics=10, num_words=3):
            topicDims.append(item[1]+"\n")
            print("\t\t",item)
        IO.writeFile(Destpath,topicDims)
        print("========================================================")


def main():

    Sourpath = os.getcwd() + "\\..\\Data\\SourceData\\Group_45494_events.csv"
    Destpath = os.getcwd() + "\\..\\Data\\DestData\\eventTopicDimsLSI.txt"
    ModelPath = os.getcwd()+"\\..\\Model\\LSI"
    # plotpath = os.getcwd() + "\\..\\Model\\eventTopics.csv"
    # plotpath = os.getcwd() + "\\..\\Model\\group45494eventTopicsR.csv"

    text = textPreprocess(Sourpath)
    text.Modeling(ModelPath,Destpath)
    # PlotTopics(plotpath)



if __name__ == "__main__":
    main()
