# %%
"""
数据来源：https://github.com/L1aoXingyu/Char-RNN-Gluon/tree/master/data
"""
import codecs
import collections
import numpy as np
import logging
import time


# %%
class TextDataLoader(object):
    def __init__(self):
        self.word_to_index = None
        self.index_to_word = None
        self.data = None
        self.data_size = 0
        self.vocab = None
        self.vocab_num = 0

    def load_data(self, file_path, max_vocab_num = 5000):
        start_time = time.time()
        with codecs.open(file_path, mode='r', encoding='utf-8') as f:
            file_content = f.readlines()

        word_list = [w for line in file_content for w in line]
        vocab_count_dict = collections.Counter(word_list)
        vocab_count_arr = list(vocab_count_dict.items())
        vocab_count_arr.sort(key=lambda x: x[1], reverse=True)
        real_vocab_num = len(vocab_count_arr)
        if len(vocab_count_arr) > max_vocab_num:
            vocab_count_arr = vocab_count_arr[:max_vocab_num]
        self.vocab = [x[0] for x in vocab_count_arr]
        self.word_to_index = {w: i for i, w in enumerate(self.vocab)}
        self.index_to_word = dict(enumerate(self.vocab))
        self.vocab_num = len(self.vocab)
        data = [ [self.word_to_index[w] for w in line if self.word_to_index.get(w)!= None] 
                for line in file_content ] 
        self.data = data
        logging.info("load data done. real_vocab_num:%d result_vocab_num:%d data_size:%d",
                     real_vocab_num, self.vocab_num, len(self.data))
        
            
        
        

# %%


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    loader = TextDataLoader()
    loader.load_data("poetry.txt", 100)
    print("word_to_index:%s" % loader.word_to_index)
    print("index_to_word:%s" % loader.index_to_word)
    print(len(loader.data))
    print(loader.data[0])
    print(loader.data[-1])
    logging.getLogger().handlers[0].flush()
