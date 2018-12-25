###################################################################
#将所有文档作为语料库，从所有的文档中提取除指定个数的主题(主题维度)；
#corpus:语料库来自Coursera的课程数据，总共379个课程，每行包括3部分内容：
#课程名\t课程简介\t课程详情, 已经清除了其中的html tag.
#Author：FlashXT;
#Date:2018.12.14,Friday;
#CopyRight © 2018-2020,FlashXT & turboMan . All Right Reserved.
###################################################################
import os
import re

from nltk import word_tokenize, RegexpTokenizer
from smart_open import smart_open
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models, similarities

# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

class MyCorpus(object):
    def __iter__(self,path):
        for line in smart_open(path, 'rb'):
            # assume there's one document per line, tokens separated by whitespace
            yield Dictionary.doc2bow(line.lower().split())

class textPreprocess():
    def __init__(self,Sourpath):
        '''
        预处理类的初始化
        :param SourPath:
        '''
        file = open(Sourpath, encoding="utf-8")
        texts = []

        for item in file:
            #去除逗号等非字母字符，每个文档为列表中的一个元素
            texts.append(re.sub(r'[^][a-z]+'," ",item.lower()).split(" "))
        self.content = texts

    def Preprocessing(self,Modelpath):

        #  去除停用词
        list = []
        #获取英语的停用词表
        en_stop = stopwords.words('english')    # get_stop_words('en')

        #去除文本中的停用词
        for item in self.content:
            stopped_tokens = [i for i in item if not i in en_stop and i !=""]
            list.append(stopped_tokens)
        self.content = list

        #  去掉低词频的词
        all_stems = sum(self.content, [])
        stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
        texts = [[stem for stem in text if stem not in stems_once] for text in self.content]
        self.content = texts

        #  生成字典和语料库

        # # corpora.Dictionary 对象,类似python中的字典对象, 其Key是字典中的词，其Val是词对应的唯一数值型ID
        dictionary = corpora.Dictionary(text for text in texts)
        # # print(dictionary.token2id)#输出word与id的对应关系
        # # dictionary.doc2bow(doc)是把文档 doc变成一个稀疏向量，[(0, 1), (1, 1)]，#表明id为0,1的词汇出现了1次。
        corpus = [dictionary.doc2bow(text) for text in texts]

        # 存储字典和语料库
        if not os.path.exists(Modelpath):
            os.mkdir(Modelpath)
        dictionary.save(os.path.join(Modelpath, 'coursera_corpus.dict'))  # store the dictionary, for future reference
        corpora.MmCorpus.serialize(os.path.join(Modelpath, 'coursera_corpus.mm'), corpus)

        return corpus,dictionary

    def Modeling(self,Modelpath):
        '''
        :return:
        '''

        if os.path.exists(os.path.join(Modelpath+"\\CorDicData", 'coursera_corpus.dict')):

            # 加载字典和语料库
            dictionary = corpora.Dictionary.load(os.path.join(Modelpath+"\\CorDicData", 'coursera_corpus.dict'))
            corpus = corpora.MmCorpus(os.path.join(Modelpath+"\\CorDicData", 'coursera_corpus.mm'))

        else :
            corpus, dictionary = self.Preprocessing(Modelpath+"\\CorDicData")
        print(corpus)
        print(dictionary)
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        modellsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=10)
        print("========================================================")
        print("The Docs Topic Dims:")
        for item in modellsi.print_topics(num_topics=10, num_words=1):
            print("\t\t", item)
        print("========================================================")
        return modellsi

    def computeSimilarity(self,model,corpus,dictionary,sourDoc):
        dictionary = corpora.Dictionary.load(dictionary)
        corpus = corpora.MmCorpus(corpus)

        corpusfile = open(sourDoc, encoding="utf-8")
        courses = [line.strip() for line in corpusfile]
        courses_name = [course.split('\t')[0] for course in courses]
        courses_info = [course.split('\t')[1] for course in courses]
        courses_content = [course.split('\t')[2] for course in courses]

        index = similarities.MatrixSimilarity(model[corpus])
        print(type(index))
        # for item in courses_content:
        #     item_bow = dictionary.doc2bow(item.split(" "))
        #     ml_lsi = model[item_bow]
        #
        #     sims = index[ml_lsi]
        #
        #     sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
        #
        #     print("========================================================")
        #     for item in sort_sims[0:10]:
        #         print(item[0],'\t\t',courses_name[item[0]],'\t\t',item[1])
        #     print("========================================================")




def main():


    Sourpath = os.getcwd()+"\\..\\Data\\SourceData\\coursera_corpus"
    ModelPath = os.getcwd()+"\\..\\Model\\Coursera"

    text = textPreprocess(Sourpath)
    model = text.Modeling(ModelPath)
    corpuspath = os.path.join(ModelPath+"\\CorDicData", 'coursera_corpus.mm')
    dicpath = os.path.join(ModelPath+"\\CorDicData", 'coursera_corpus.dict')

    text.computeSimilarity(model,corpuspath,dicpath,Sourpath)


if __name__ == "__main__":
    main()
