import os
import re
import warnings
from ToolClasses import IO
import matplotlib.pyplot as plt
warnings.filterwarnings(action='ignore')


#计算两个文档的相似度
def topicSimilarity(etopicpath,gtopicpath):

    path = os.getcwd() + "\\..\\Data\\DestData\\EventCountbyDimsLDA.csv"

    etopics = IO.csv_reader(etopicpath)
    gtopics = IO.csv_reader(gtopicpath)

    list =[]
    for gt in gtopics:
        count = 0
        gtopic =" ".join(re.findall(r'(?<=\*").*?(?=")', gt[1]))
        for item in etopics:
            if item[2] == gtopic:
                count += 1
        list.append([gt[0],gtopic,count])

    list.insert(0, ['TopicNo','Topic','Count'])
    IO.csv_writer(path, list)

def plot(filepath):
    file = IO.csv_reader(filepath)
    etopics =[]
    [etopics.append([i[0],i[2]]) for i in file]

    x = []
    y = []
    for item in etopics:
        x.append(item[0])
        y.append(int(item[1]))

    # 添加图形属性
    fig = plt.figure(figsize=(10.0, 10.0))
    ax = plt.subplot()
    ax.bar(x, y, facecolor='green', width=0.8)

    plt.xlabel('Topics')
    plt.ylabel('TopicsCount')
    plt.title('The Statistics of Group45494 Events Topics')
    plt.show()


def main():

    etopicpath = os.getcwd()+"\\..\\Data\\DestData\\Group45494eventTopic.csv"
    gtopicpath = os.getcwd() + "\\..\\Data\\DestData\\eventTopicDimsLDA.csv"
    plotpath = os.getcwd()+"\\..\\Data\\DestData\\EventCountbyDimsLDA.csv"

    topicSimilarity(etopicpath,gtopicpath)
    plot(plotpath)


if __name__ == "__main__":
    main()