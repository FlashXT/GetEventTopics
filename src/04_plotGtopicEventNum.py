import os

from ToolClasses import IO
import matplotlib.pyplot as plt


def plot(list):
    x = []
    y = []
    for item in list:
        x.append(item[0][1])
        y.append(int(item[1]))

    # 添加图形属性
    fig = plt.figure(figsize=(17.0, 17.0))
    ax = plt.subplot()
    ax.bar(x, y, facecolor='green', width=0.8)

    plt.xlabel('Topics')
    plt.ylabel('TopicsCount')
    plt.title('The Statistics of Group45494 Events Topics')

    # plt.bar(x, y, facecolor='green', width=0.8)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    plt.show()

def main():
    dims2Tpath = os.getcwd()+"\\..\\Data\\DestData\\Dims2Gtopics.csv"
    eventCountpath = os.getcwd()+"\\..\\Data\\DestData\\EventCountbyDimsLDA.csv"
    dims2T = IO.csv_reader(dims2Tpath)
    # print(dims2T[0])
    eventInfo = IO.csv_reader(eventCountpath)
    # print(eventInfo[0])

    topics =[]
    i = 0
    while i < len(eventInfo):
        topics.append([[dims2T[i][2],dims2T[i][3]],eventInfo[i][2]])
        i+=1

    list =[]
    [list.append(item[0]) for item in topics if item[0] not in list]
    # [print(l) for l in list]
    text =[]
    for item in list:
        count = 0
        for topic in topics:
            if item == topic[0]:
                count = count +int(topic[1])
        text.append([item,count])
    [print(item) for item in text]
    plot(text)
    # topics.insert(0, ["topicId", "topic", 'Num'])
    # for item in topics:
    #     print(item)


if __name__ == "__main__":
    main()