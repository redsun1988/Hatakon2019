import json
import datetime

class ModelSerializer:
    @classmethod
    def getCurrentDateAsString(self):
        return datetime.datetime.now().strftime("%B_%d_%Y_%I-%M%p")

    @staticmethod
    def saveMetaInfoToJSON(vocabulary, reverse_token_index, token_index, 
    max_input_len, max_output_len, vocab_size):
        data = {}  
        data['vocabulary'] = vocabulary
        data['reverse_token_index'] = reverse_token_index
        data['token_index'] = token_index
        data['max_input_len'] = int(max_input_len)
        data['max_output_len'] = int(max_output_len)
        data['vocab_size'] = int(vocab_size)

        
        path = "./data/metaParams"+ ModelSerializer.getCurrentDateAsString()+".json"
        with open(path, 'w') as outfile:  
            json.dump(data, outfile)

