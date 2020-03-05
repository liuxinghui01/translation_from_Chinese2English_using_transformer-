from pathlib import Path
import os
import tensorflow as tf
import numpy as np

def read_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        words = []
        for line in f:
            line =line.strip()
            words += line.split(' ')
    return words

def build_vocab(words):
    words_dict = {}
    for word in words:
        if word in words_dict:
            words_dict[word] += 1
        else:
            words_dict[word] = 1
    # d= words_dict.items()
    dic = sorted(words_dict.items(), key=lambda d:d[1], reverse=True)
    word2id = {}
    id2word = {}
    word2id['Unknown'] = 0
    id2word[0] = 'Unknown'
    word2id['SOS'] = 1
    id2word[1] = 'SOS'
    word2id['EOS'] = 2
    id2word[2] = 'EOS'
    #  0 对应的是not a num
    for i, item in enumerate(dic):
        word2id[item[0]] = i+3
        id2word[i+3] = item[0]
    # words_dict = 1
    return word2id, id2word
# 将句子按空格切词后再按照字典word2id将其映射成数字串
def sent2mat(sent_path, word2id_chinese, sentence_length):
    with open(sent_path, 'r', encoding='utf-8') as f:
        input_x = []
        for line in f:
            line = line.strip()
            line_words = line.split(' ')
            # 将每一行放到tmp中去
            # 1表示"SOS"
            tmp = [1]
            for word in line_words:
                tmp.append(word2id_chinese.get(word, 0))
            # 长的句子截断，短的句子补零
            if len(tmp)>=sentence_length:
                tmp = tmp[:sentence_length]
            else:
                tmp = tmp + [0]*(sentence_length-len(tmp))
            tmp.append(2)
            print(len(tmp))
            assert len(tmp) == sentence_length + 1
            input_x.append(tmp)
    return input_x

# if __name__ == '__main__':
def data_processor():
    path_chinese = os.path.join(os.path.abspath('.'), 'chinese-english-corpus\\chinese1.txt')
    path_english = os.path.join(os.path.abspath('.'), 'chinese-english-corpus\\english1.txt')

    words_dict_chinese = read_data(path_chinese)
    words_dict_english = read_data(path_english)

    word2id_chinese, id2word_chinese = build_vocab(words_dict_chinese)
    word2id_english, id2word_english = build_vocab(words_dict_english)

    # 将每行表示成数字
    sentence_length = 50
    input_x = sent2mat(path_chinese, word2id_chinese, sentence_length)
    input_y = sent2mat(path_english, word2id_english, sentence_length)

    print('input_x', np.array(input_x).shape)
    print('input_y', np.array(input_y).shape)

    ds_train = tf.data.Dataset.from_tensor_slices((input_x, input_y))
    ds_train = ds_train.shuffle(300)
    ds_train = ds_train.batch(30)
    return ds_train, len(word2id_chinese), len(word2id_english)




    # end=1