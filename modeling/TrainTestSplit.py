def TrainTestSplit(data, labels, testSize):
    num_examples = len(data)
    split_index = int(num_examples * testSize)

    dataTest = data[split_index:]
    dataTrain = data[:split_index]

    labelsTrain = labels[split_index:]
    labelsTest = labels[:split_index]
    return dataTrain, labelsTrain, dataTest, labelsTest