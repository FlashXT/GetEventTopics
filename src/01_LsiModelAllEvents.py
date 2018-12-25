###############################################################################
# ① 训练模型：（(event name,description)数据 ---> 模型）
#   将所有事件的(event name,description)作为语料库，使用LDA模型从所有的文档中提取除指定个
#   数的主题维度；
# ② 解释数据：（模型 --->event Topic）
#   使用训练出模型主题维度对event 进行表示；
#Author：FlashXT;
#Date:2018.12.20,Thursday;
#CopyRight © 2018-2020,FlashXT & turboMan . All Right Reserved.
###############################################################################
import os
import re
from ToolClasses import IO
from smart_open import smart_open
from gensim.corpora import Dictionary
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from nltk.corpus import stopwords
from gensim import corpora, models
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

class MyCorpus(object):
    def __iter__(self,path):
        for line in smart_open(path, 'rb'):
            # assume there's one document per line, tokens separated by whitespace
            yield Dictionary.doc2bow(line.lower().split())

class textPreprocess():
    def __init__(self,SourPath):
        '''
        预处理类的初始化
        :param SourPath:
        :param DestPath:
        '''
        self.SourPath = SourPath
        self.content = IO.csv_reader(SourPath)
        list = []
        for item in self.content:
            list.append([item[3],item[10]])
        self.content = list

    def processText(self,text):

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
        # temp2 = [wnl.lemmatize(i, pos='v') for i in content]
        # [temp1.append(i) for i in temp2 if i not in temp1]

        content = temp1
        # print(content)

        # ⑥ 去掉低词频的词
        # all_stems = sum(self.content, [])
        # stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
        # texts = [[stem for stem in text if stem not in stems_once] for text in self.content]
        # self.content = texts
        return content

    def Preprocessing(self,Modelpath):
        texts = []
        for item in self.content:
            texts.append(self.processText(item))

        # 生成字典和语料库
        # corpora.Dictionary 对象
        # 类似python中的字典对象, 其Key是字典中的词，其Val是词对应的唯一数值型ID
        dictionary = corpora.Dictionary(item for item in texts)
        # print(dictionary.token2id)

        # dictionary.doc2bow(doc)是把文档 doc变成一个稀疏向量，[(0, 1), (1, 1)]，
        # 表明id为0,1的词汇出现了1次。 \
        corpus = [dictionary.doc2bow(item) for item in texts]
        # tfidf = models.TfidfModel(corpus)
        # corpus_tfidf = tfidf[corpus]

        # 存储字典和语料库
        if not os.path.exists(Modelpath):
            os.mkdir(Modelpath)
        dictionary.save(os.path.join(Modelpath,'group45494.dict'))  # store the dictionary, for future reference
        corpora.MmCorpus.serialize(os.path.join(Modelpath,'group45494.mm'), corpus)

        return corpus,dictionary

    def LSIModeling(self,Modelpath,Destpath):
        '''
        :return:
        '''
        if os.path.exists(os.path.join(Modelpath+"\\Model",'LsiModel.mdl')):
            lsimodel = models.LsiModel.load(os.path.join(Modelpath+"\\Model",'LsiModel.mdl'))
        else:
            if os.path.exists(os.path.join(Modelpath+"\\CorDicData", 'group45494.dict')):

                # 加载字典和语料库
                dictionary = corpora.Dictionary.load(os.path.join(Modelpath+"\\CorDicData", 'group45494.dict'))
                corpus = corpora.MmCorpus(os.path.join(Modelpath+"\\CorDicData", 'group45494.mm'))
            else :
                corpus, dictionary = self.Preprocessing(Modelpath+"\\CorDicData")
            lsimodel = models.lsimodel.LsiModel(corpus, num_topics=10, id2word=dictionary)
            lsimodel.save(os.path.join(Modelpath+"\\Model", 'LsiModel.mdl'))

        topicDims =[]
        print("========================================================")
        print("The Docs Topic Dims:")
        for item in lsimodel.print_topics(num_topics=10):
            topicDims.append([item[0],item[1]])
            print("\t\t", item)
        topicDims.insert(0,['no','topic'])
        IO.csv_writer(Destpath, topicDims)
        print("========================================================")

    def GetEventTopic(self,modelpath,textspath,dictpath,topicspath):
        dictionary = corpora.Dictionary.load(dictpath)
        model = models.LsiModel.load(modelpath)
        texts = IO.csv_reader(textspath)
        events =[]
        [events.append((event[1],[event[3],event[10]])) for event in texts]

        topics = []
        for event in events:

            text = self.processText(event[1])
            event_bow = dictionary.doc2bow(text)
            event_lda = model[event_bow]  # 得到新文档的主题分布

            temp = (0,0)
            for item in event_lda:
                if item[1] > temp[1]:
                    temp =item
            items = re.findall(r'(?<=\*").*?(?=")', model.print_topic(temp[0]))
            topics.append([event[0],event[1][0]," ".join(items)])
            print(event[0], event[1][0], items)

        topics.insert(0,["event id","event name","event topic"])
        IO.csv_writer(topicspath,topics)

            # for item in items:
            #     print(item)
            # print("%s\t%f" % (model.print_topic(temp[0]),temp[1]))
        print("save eventTopics Finished!")


def main():

    Sourpath = os.getcwd()+"\\..\\Data\\SourceData\\Group_45494_events.csv"
    Destpath = os.getcwd()+"\\..\\Data\\DestData\\eventTopicDimsLSI.csv"
    ModelPath = os.getcwd()+"\\..\\Model\\LSI"
    corpuspath = os.getcwd()+"\\..\\Model\\LSI\\CorDicData\\group45494.mm"
    dictpath = os.getcwd()+"\\..\\Model\\LSI\\CorDicData\\group45494.dict"
    topicspath = os.getcwd() + "\\..\\Data\\DestData\\Group45494eventTopicLSI.csv"

    text = textPreprocess(Sourpath)
    text.LSIModeling(ModelPath,Destpath)

    text.GetEventTopic(ModelPath+"\\Model\\LsiModel.mdl", Sourpath,dictpath,topicspath)


if __name__ == "__main__":
    main()
