#This file is used to generate new RNR models and save them to files
from TrainTestSplit import TrainTestSplit
from DataGenerator import TrainTestSplit

import pandas as pd
import numpy as np
import re
import os
import tensorflow as tf
import nltk
import json
import datetime

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import TensorBoard, LambdaCallback

from modeling.data import read_csv

class ModelBuilder:

    def __init__(self, csfFilePath, originTextColumnName, targetTextColumnName, percent):
       #'./data/SubjectsQuestionsAllExtended.csv', 'Text', 'Subject', 75
       data, labels, max_input_len, max_output_len = read_csv(csfFilePath, originTextColumnName, targetTextColumnName, percent)
       self.data = data
       self.labels = labels
       self.max_input_len = max_input_len
       self.max_output_len = max_output_len
       
       self.vocabulary = sorted(list(set(self.uniqueChars(data) + self.uniqueChars(labels))))
       self.token_index = dict([(char, i) for i, char in enumerate(self.vocabulary)])
       self.reverse_token_index = dict([(i, char) for char, i in self.token_index.items()])

       #Some metaparameters for learning
       self.batch_size = 72
       self.vocab_size = len(self.vocabulary)
       self.latent_dim = 256
       
       self.saveMetaInfoToJSON()

    def uniqueChars(self, p_list):
       return list(set((''.join([''.join(set(p)) for p in p_list]))))

    def saveMetaInfoToJSON(self):
        data = {}  
        data['vocabulary'] = self.vocabulary
        data['reverse_token_index'] = self.reverse_token_index
        data['token_index'] = self.token_index
        data['max_input_len'] = int(self.max_input_len)
        data['max_output_len'] = int(self.max_output_len)
        data['vocab_size'] = int(self.vocab_size)

        
        path = "./data/metaParams"+self.getCurrentDateAsString()+".json"
        with open(path, 'w') as outfile:  
            json.dump(data, outfile)

    def getCurrentDateAsString(self):
        return datetime.datetime.now().strftime("%B_%d_%Y_%I-%M%p")

    def __BuildGenerators(self):
        dataTrain, labelsTrain, dataTest, labelsTest = TrainTestSplit(self.data, self.labels);
        self.train_generator = DataGenerator(dataTrain, labelsTrain)
        self.valid_generator = DataGenerator(dataTest, labelsTest)