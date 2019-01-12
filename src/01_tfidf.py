import logging
import os
import re

from nltk import RegexpTokenizer

from ToolClasses import IO
from string import punctuation
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
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

        tokenizer = RegexpTokenizer(r'[a-z]+')
        raw = str(content).lower()
        content = tokenizer.tokenize(raw)
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
        stopped_tokens = [i for i in content if not i in en_stop]
        # print(stopped_tokens)
        corpus.append([item[0], item[3], stopped_tokens])

    return corpus

def textTFIDF(texts):

    # 得到文档-单词矩阵 （直接利用统计词频得到特征）
    # dictionary = corpora.Dictionary(texts)  # 得到单词的ID,统计单词出现的次数以及统计信息
    # # print(dictionary.token2id)  # 可以得到单词的id信息  <dict>
    # texts = [dictionary.doc2bow(text) for text in texts]  # 将dictionary转化为一个词袋，得到文档-单词矩阵
    # # print(type(dictionary))  # 得到的是gensim.corpora.dictionary.Dictionary的class类型
    # # 利用tf-idf来做为特征进行处理
    # texts_tf_idf = models.TfidfModel(texts)[texts]     # 文档的tf-idf形式(训练加转换的模式)
    # for text in texts_tf_idf:            # 逐行打印得到每篇文档的每个单词的TD-IDF的特征值
    #     print(text)


    tfidf_vectorizer = TfidfVectorizer()
    texts_tf_idf = tfidf_vectorizer.fit_transform(texts)
    print(tfidf_vectorizer.get_feature_names())
    print(tfidf_vectorizer.vocabulary_)
    print(texts_tf_idf.toarray())
    # for item in texts_tf_idf:
    #     print(item)
    print("AAAAAAAA")
    # print(tfidf_vectorizer.idf_)                    # 特征对应的权重
    # print(tfidf_vectorizer.get_feature_names())     # 特征词
    # print(real_vec.toarray())                       # 语料库对应的向量表示

    return texts_tf_idf


def main():
    textpath = os.getcwd() + "\\..\\Data\\SourceData\\Group_45494_events.csv"
    modelpath = os.getcwd()+"\\..\\Model\\Word2Vec\\Word2vecmodel.model"
    corpus = getCorpus(textpath)
    # print(corpus[0])
    texts =[" ".join(item[2]) for item in corpus]
    texts_tf_idf = textTFIDF(texts)
    # model = Word2Vec(sentences = texts_tf_idf, size=100, window=5, min_count=1, workers=2)
    # model.save(modelpath)
    # print(model["away"])

if __name__ == "__main__":
    main()