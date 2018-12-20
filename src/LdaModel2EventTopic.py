############################################################################
# 训练好的LDA模型从语料库中提取出了指定个数的主题维度；那么语料库中的每一篇文章如何利用得到的
# 主题维度表示出来？即模型主题维度的表示能力；
#
#Author：FlashXT;
#Date:2018.12.19,Wednesday;
#CopyRight © 2018-2020,FlashXT & turboMan. All Right Reserved.
############################################################################
import os
from gensim import corpora, models

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

def CorpraDimsRepresentation(corpuspath,dicpath,modelpath):

    dictionary = corpora.Dictionary.load(dicpath)
    corpus = corpora.MmCorpus(corpuspath)
    ldamodel = models.LdaModel.load(modelpath)
    print(dictionary.token2id)
    # for item in corpus:
    print(ldamodel)





def main():
    ROOTPATH = os.getcwd()+"\\..\\Model\\LDA"
    corpuspath = os.path.join(ROOTPATH+"\\CorDicData",'group45494.mm')
    dicpath = os.path.join(ROOTPATH + "\\CorDicData", 'group45494.dict')
    Destpath = os.getcwd() + "\\..\\Data\\DestData\\eventTopicDimsLDA.txt"
    modelpath = os.path.join(ROOTPATH+"\\Model",'LdaModel.mdl')

    CorpraDimsRepresentation(corpuspath, dicpath,modelpath)


if __name__ == "__main__":
    main()
