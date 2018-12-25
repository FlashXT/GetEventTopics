
import os
import IO
import gensim


def readData(path1,path2):
    '''
    :param path1:
    :param path2:
    :return:
    '''
    data1 = IO.csv_reader(path1)
    data2 = IO.csv_reader(path2)
    eventTopics = []
    [eventTopics.append(i[1].lower()) for i in data1]
    groupTopics = []
    [groupTopics.append(i[3].lower()) for i in data2]
    return eventTopics,groupTopics


def computeSimilarity():

    modelpath = os.getcwd()+"\\..\\data\\GoogleNews-vectors-negative300.bin"
    model = gensim.models.KeyedVectors.load_word2vec_format(modelpath, binary=True)

    path1 = os.getcwd() + "\\..\\data\\group45494EventsTopics.csv"
    path2 = os.getcwd() + "\\..\\data\\Group_45494_topics1.csv"
    eventTopics, groupTopics = readData(path1, path2)
    path = os.getcwd() + "\\..\\data\\WordSimilarity.csv"
    list = []
    for item in eventTopics:
        sim = {'key': None, 'value': 0}
        for topic in groupTopics:
            if sim['value'] < model.n_similarity([item], [topic]):
                sim['key'] = topic
                sim['value'] = model.n_similarity([item],[topic])

        list.append([item, sim['key'], sim['value']])
        print(str(item + '<->' + sim['key']), '\t', sim['value'])
        print("================================================")

    list.insert(0, ['eventTopic', 'groupTopic', 'Similarity'])
    IO.csv_writer(path, list)
    # for item in eventTopics:
    #     for topic in groupTopics:
    #         print(item,topic,model.n_similarity([item],[topic]))

def main():
    computeSimilarity()

if __name__=="__main__":
    main()