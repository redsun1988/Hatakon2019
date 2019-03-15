import numpy as np
from keras.models import load_model
import pandas as pd
import re
import json

class TextGenerator:
    def __init__(self, encoderModelPath, decoderModelPath, hyperParamsFilePath):
        self.encoder_model = load_model(encoderModelPath)
        self.decoder_model = load_model(decoderModelPath)
        self.load_data(hyperParamsFilePath)

    def load_data(self, hyperParamsFilePath):
        with open(hyperParamsFilePath) as json_file:  
           data = json.load(json_file)
        self.max_input_len = data['max_input_len']
        self.max_output_len = data['max_output_len']
        self.token_index = data['token_index']
        self.reverse_token_index = dict(
            (i, char) for char, i in self.token_index.items())
        self.vocabulary = data['vocabulary']
        self.vocab_size = len(self.vocabulary)

    def decode_sequence(self, input_seq):
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.vocab_size))
        # Populate the first character of target sequence with the start character.
        # target_seq[0, 0, token_index['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_token_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
                    len(decoded_sentence) > self.max_output_len):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.vocab_size))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence

    def encode_string(self, text):
        encoder_input_data = np.zeros(
            (1, self.max_input_len, self.vocab_size),
            dtype='float32')

        input_len = len(text)

        for t in range(self.max_input_len):
            char = text[t] if t < input_len else ' '
            encoder_input_data[0, t, self.token_index[char]] = 1

        return encoder_input_data

    def predict(self, text):
        input_seq = self.encode_string(text)
        decoded_sentence = self.decode_sequence(input_seq)

        print('-----------Predict----------')
        print('Input sentence:', text)
        print('Decoded sentence:', decoded_sentence)

        return decoded_sentence
