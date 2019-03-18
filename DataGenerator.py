import numpy as np

class DataGenerator:
    def __init__(self, pars, heads, 
    batch_size, vocab_size, max_input_len, max_output_len, token_index):
        self.pars = pars
        self.heads = heads
        self.max = len(pars) - 1
        self.n = 0
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.max_input_len = max_input_len
        self.token_index = token_index
        self.max_output_len = max_output_len

    def __next__(self):
        start = self.n
        end = self.n + self.batch_size
        
        encoder_input_data = np.zeros(
            (self.batch_size, self.max_input_len, self.vocab_size),
            dtype='float32')
        decoder_input_data = np.zeros(
            (self.batch_size, self.max_output_len, self.vocab_size),
            dtype='float32')
        decoder_target_data = np.zeros(
            (self.batch_size, self.max_output_len, self.vocab_size),
            dtype='float32')
        
        batch_paragraths = self.pars[start: end]
        batch_headlines = self.heads[start: end]
        
        for i, (input_text, target_text) in enumerate(zip(batch_paragraths, batch_headlines)):
            input_len = len(input_text)
            target_len = len(target_text)
            for t in range(input_len):
                char = input_text[t]
                encoder_input_data[i, t, self.token_index[char]] = 1
            for t in range(target_len):
                char = target_text[t]
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t, self.token_index[char]] = 1.
                if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                    decoder_target_data[i, t - 1, self.token_index[char]] = 1
                    
        self.n = self.n + 1
        
        if self.n > self.max:
            self.n = 0
        
        return ([encoder_input_data, decoder_input_data], decoder_target_data)