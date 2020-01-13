# -*- coding: UTF-8 -*-
#

"""
数据来源：https://github.com/L1aoXingyu/Char-RNN-Gluon/tree/master/data
"""
import codecs
import collections


class TextDataLoader(object):
    def __init__(self):
        self.word_to_index = None
        self.index_to_word = None
        self.vocab = None

    def load_data(self, file_path, max_vocab_num = 5000):

        with codecs.open(file_path, mode='r', encoding='utf-8') as f:
            file_content = f.readlines()

        word_list = [w for line in file_content for w in line]
        vocab_count_dict = collections.Counter(word_list)
        vocab_count_arr = list(vocab_count_dict.items())
        vocab_count_arr.sort(key=lambda x: x[1], reverse=True)
        if len(vocab_count_arr) > max_vocab_num:
            vocab_count_arr = vocab_count_arr[:max_vocab_num]
        self.vocab = [x[0] for x in vocab_count_arr]
        self.word_to_index = {w: i for i, w in enumerate(self.vocab)}
        self.index_to_word = dict(enumerate(self.vocab))




if __name__ == "__main__":
    loader = TextDataLoader()
    loader.load_data("poetry.txt", 10)
    print("word_to_index:%s" % loader.word_to_index)
    print("index_to_word:%s" % loader.index_to_word)
