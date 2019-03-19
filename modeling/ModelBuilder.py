#This file is used to generate new RNR models and save them to files
from TrainTestSplit import TrainTestSplit
from DataGenerator import DataGenerator
from StringProcessor import StringProcessor

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
       data, labels, max_input_len, max_output_len = read_csv(csfFilePath, originTextColumnName, targetTextColumnName, percent)
       self.data = data
       self.labels = labels
       self.max_input_len = max_input_len
       self.max_output_len = max_output_len
       
       self.vocabulary = StringProcessor.GetUniqueChars(list(data) + list(labels)) 
       self.token_index = StringProcessor.GetReversTokenIndex(self.vocabulary) 
       self.reverse_token_index = StringProcessor.GetReversTokenIndex(self.vocabulary) 

       #Some metaparameters for learning
       self.batch_size = 72
       self.vocab_size = len(self.vocabulary)
       self.latent_dim = 256
       
       self.saveMetaInfoToJSON()
       self.__BuildGenerators()
       self.__InitModel()

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
        dataTrain, labelsTrain, dataTest, labelsTest = TrainTestSplit(self.data, self.labels, .95)
        self.train_generator = DataGenerator(dataTrain, labelsTrain, self.batch_size, self.vocab_size, self.max_input_len, self.max_output_len, self.token_index)
        self.valid_generator = DataGenerator(dataTest, labelsTest, self.batch_size, self.vocab_size, self.max_input_len, self.max_output_len, self.token_index)

    def __InitModel(self):
      ##Build model
      
      # Define an input sequence and process it.
      encoder_inputs = Input(shape=(None, self.vocab_size))
      encoder = LSTM(self.latent_dim, return_state=True)
      encoder_outputs, state_h, state_c = encoder(encoder_inputs)
      # We discard `encoder_outputs` and only keep the states.
      encoder_states = [state_h, state_c]
      
      # Set up the decoder, using `encoder_states` as initial state.
      decoder_inputs = Input(shape=(None, self.vocab_size))
      # We set up our decoder to return full output sequences,
      # and to return internal states as well. We don't use the
      # return states in the training model, but we will use them in inference.
      decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
      decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                           initial_state=encoder_states)
      decoder_dense = Dense(self.vocab_size, activation='softmax')
      decoder_outputs = decoder_dense(decoder_outputs)
      
      # Define the model that will turn
      # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
      self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
      
      fname = "./data/model_weights"+self.getCurrentDateAsString()+".h5"
      
      if os.path.isfile(fname):
          self.model.load_weights(fname)
      
      # Run training
      
      adam_optimiser = Adam(lr=0.0016, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
      
      self.model.compile(optimizer=adam_optimiser, loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self):
        self.model.fit_generator(self.train_generator,
                    steps_per_epoch=5,
                    epochs=1,
                    validation_data=self.valid_generator,
                    validation_steps=10,
                    callbacks=[])

    def ShowModelSummary(self):
      self.model.summary()

#region Tests (kind of tests ^_^ )
if __name__ == "__main__":
   #region ModelBuilder can be created test
   mBuilder = ModelBuilder('./data/SubjectsQuestions100k.csv', 'Text', 'Subject', 75)
   #endregion

   #region ModelBuilder fit test
   mBuilder.fit()
   #endregion

#endregion