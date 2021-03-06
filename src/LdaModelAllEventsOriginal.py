############################################################################
# 将所有事件的(event name,description)作为语料库，使用LDA模型从所有的文档中提取除指定个
# 数的主题维度；
#
#Author：FlashXT;
#Date:2018.12.9,Sunday;
#CopyRight © 2018-2020,FlashXT & turboMan . All Right Reserved.
############################################################################
import os
import re
import nltk
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

    def Preprocessing(self,Modelpath):

        # ① 去除HTML标签

        list = []
        for item in self.content:
            content = re.sub(r'<[^>]*>', ' ', item[1])
            list.append((item[0] + " " + content).lower())
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
        for item in self.content:
            stopped_tokens = [i for i in item if not i in en_stop]
            # stopped_tokens = " ".join(i for i in stopped_tokens)
            list.append(stopped_tokens)
        self.content = list

        # ④ 按长度过滤
        list = []
        for item in self.content:
            temp = [i for i in item if len(i) > 3]
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
        # all_stems = sum(self.content, [])
        # stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
        # texts = [[stem for stem in text if stem not in stems_once] for text in self.content]
        # self.content = texts

        # ⑦ 生成字典和语料库
        # corpora.Dictionary 对象
        # 类似python中的字典对象, 其Key是字典中的词，其Val是词对应的唯一数值型ID
        dictionary = corpora.Dictionary(item for item in list)
        # print(dictionary.token2id)

        # dictionary.doc2bow(doc)是把文档 doc变成一个稀疏向量，[(0, 1), (1, 1)]，
        # 表明id为0,1的词汇出现了1次。 \
        corpus = [dictionary.doc2bow(item) for item in list]
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]

        # ⑧ 存储字典和语料库
        if not os.path.exists(Modelpath):
            os.mkdir(Modelpath)
        dictionary.save(os.path.join(Modelpath,'group45494.dict'))  # store the dictionary, for future reference
        corpora.MmCorpus.serialize(os.path.join(Modelpath,'group45494.mm'), corpus_tfidf)

        return corpus,dictionary

    def LDAModeling(self,Modelpath,Destpath):
        '''
        :return:
        '''
        if os.path.exists(os.path.join(Modelpath+"\\Model",'LdaModel.mdl')):
            ldamodel = models.LdaModel.load(os.path.join(Modelpath+"\\Model",'LdaModel.mdl'))
        else:
            if os.path.exists(os.path.join(Modelpath+"\\CorDicData", 'group45494.dict')):

                # 加载字典和语料库
                dictionary = corpora.Dictionary.load(os.path.join(Modelpath+"\\CorDicData", 'group45494.dict'))
                corpus = corpora.MmCorpus(os.path.join(Modelpath+"\\CorDicData", 'group45494.mm'))
            else :
                corpus, dictionary = self.Preprocessing(Modelpath+"\\CorDicData")
            ldamodel = models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=5000)
            ldamodel.save(os.path.join(Modelpath+"\\Model", 'LdaModel.mdl'))

        topicDims =[]
        print("========================================================")
        print("The Docs Topic Dims:")
        for item in ldamodel.print_topics(num_topics=10, num_words=3):
            topicDims.append(item[1] + "\n")
            print("\t\t", item)
        IO.writeFile(Destpath, topicDims)
        print("========================================================")



def main():
    ROOTPATH = os.getcwd()+"\\..\\Data"
    Sourpath = os.path.join(ROOTPATH+"\\SourceData\\",'Group_45494_events.csv')
    Destpath = os.getcwd() + "\\..\\Data\\DestData\\eventTopicDimsLDA.txt"
    ModelPath = os.getcwd()+"\\..\\Model\\LDA"

    text = textPreprocess(Sourpath)
    text.LDAModeling(ModelPath,Destpath)



if __name__ == "__main__":
    main()
