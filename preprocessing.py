# -*- coding:utf-8 -*-
import codecs
import os
import re
import jieba
import random


'''
This module is for data pre-processing
'''


class DocInfo:
    def __init__(self):
        self.name = ""
        self.date = ""
        self.publish = ""
        self.title = ""
        self.asc_title = ""


def is_repeat(docs, d):
    flag = False
    old = ""
    new = ""
    for doc in docs:
        if doc.date == d.date and doc.title == d.title:
            flag = True
            old = doc.name
            new = d.name
            return flag, old, new
    return flag, old, new


# Stage 1: Remove repeated documents in data directory
def clean_docs():

    docs_dir = "./Data"
    output_dir = "./cleaned_data/stage_1_out"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print("Removing repeated documents....")
    docs_info = []
    for root, dirs, files in os.walk(docs_dir):
        for file in files:
            doc_path = os.path.join(root, file)
            doc = codecs.open(doc_path, 'r', 'utf-8')
            d_content = []
            d_info = DocInfo()
            d_info.name = doc_path
            for line in doc:
                if line.startswith("<日期>"):
                    d_info.date = line.split('=')[1]
                elif line.startswith("<版次>"):
                    d_info.publish = line.split('=')[1]
                elif line.startswith("<标题>"):
                    d_info.title = line.split('=')[1]
                elif line.startswith("<副标题>"):
                    d_info.asc_title = line.split('=')[1]
                elif line.startswith('\n') or \
                        line.startswith("<版名>") or \
                        line.startswith("<正文>") or \
                        line.startswith("<作者>") or \
                        line.startswith("<数据库>"):
                    pass
                else:
                    d_content.append(line)

            flag, old, new = is_repeat(docs_info, d_info)
            if not flag:
                docs_info.append(d_info)
                out_file = str(d_info.date[:-1])+'_'+str(d_info.publish[:-1])+'_'+str(d_info.title[:4])+'.txt'
                with codecs.open(output_dir+'/'+out_file, 'w', 'utf-8') as out_file:
                    out_file.write(str(d_info.title))
                    out_file.write(str(d_info.asc_title))
                    for content in d_content:
                        out_file.write(content)
            else:
                print("Repeated documents:", old, new)
    print("Stage 1 finished.\n")


# Stage 2: Clean non Zh-CN characters in documents
def clean_chars():

    input_dir = "./cleaned_data/stage_1_out"
    output_dir = "./cleaned_data/stage_2_out"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print("Cleaning non Zh-CN characters...")
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            doc_path = os.path.join(root, file)
            out_file = output_dir+'/'+file
            fin = codecs.open(doc_path, 'r', 'utf-8')
            fout = codecs.open(out_file, 'w', 'utf-8')
            for line in fin:
                # using range of Zh-CN coding: \u4e00 - \u9fa5
                p2 = re.compile('[^\u4e00-\u9fa5]')
                zh = " ".join(p2.split(line)).strip()
                zh = " ".join(zh.split())
                fout.write(zh.strip() + '\n')
            fin.close()
            fout.close()
    print("Stage 2 finished.\n")


# Stage 3: Participle and remove stop words
def participle():

    stopwords_path = "./setting/stop_words.txt"
    user_dicts = "./setting/user_defined_dicts.txt"
    jieba.load_userdict(user_dicts)
    input_dir = "./cleaned_data/stage_2_out"
    output_dir = "./cleaned_data/stage_3_out"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # remove stop words
    print("Participle and remove stop words...")
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            doc_path = os.path.join(root, file)
            out_file = output_dir+'/'+file
            fin = codecs.open(doc_path, 'r', 'utf-8')
            fout = codecs.open(out_file, 'w', 'utf-8')
            f_stop = codecs.open(stopwords_path, 'r', 'utf-8')

            texts = fin.read()
            stop_text = f_stop.read()
            word_list = []
            seg_list = jieba.cut(texts)
            seg_list = "/".join(seg_list)
            stop_seg_list = stop_text.split('\n')

            for word in seg_list.split('/'):
                if not (word.strip() in stop_seg_list) and len(word.strip()) > 1:
                    word_list.append(word)
            word_list = " ".join(word_list)
            fout.write(word_list)
    print("Stage 3 finished.")


# Stage 4: Remove low-frequency words in global corpus
def remove_lf_words():

    lf_words_path = "./setting/low_frequency_words.txt"
    input_dir = "./cleaned_data/stage_3_out"
    output_dir = "./cleaned_data/stage_4_out"
    word_set = set()
    freq_dict = dict()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # calculate word frequency
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            doc_path = os.path.join(root, file)
            with codecs.open(doc_path, 'r', 'utf-8') as doc:
                for line in doc:
                    for word in line.split():
                        if word not in word_set:
                            word_set.add(word)
                            freq_dict[word] = 1
                        else:
                            freq_dict[word] += 1
    print("Total unique words: ", len(freq_dict))

    # remove low-frequency words according to the setting file "low_frequency_words.txt"
    lf_words = set()
    lf_setting = codecs.open(lf_words_path, 'r', 'utf-8')
    lf_threshold = 1
    for line in lf_setting:
        if line.startswith("low_frequency_threshold"):
            lf_threshold = int(line.split('=')[1])
    print("Removing low-frequency words <= ", lf_threshold)

    for (k, v) in freq_dict.items():
        if v <= lf_threshold:
            lf_words.add(k)
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            doc_path = os.path.join(root, file)
            out_file = output_dir+'/'+file
            fin = codecs.open(doc_path, 'r', 'utf-8')
            fout = codecs.open(out_file, 'w', 'utf-8')
            word_list = fin.read()
            store_words = []
            for word in word_list.split(" "):
                if not (word.strip() in lf_words) and len(word.strip()) > 1:
                    store_words.append(word)
            store_words = " ".join(store_words)
            fout.write(store_words)
    print("Stage 4 finished.\n")


# Divide cleaned data into training data and test data
def divide_corpus():

    corpus_dir = "./cleaned_data/stage_4_out"
    lda_model_dir = "./models/lda"

    if not os.path.exists(lda_model_dir):
        os.mkdir(lda_model_dir)

    corpus =[]
    for root, dirs, files in os.walk(corpus_dir):
        for file in files:
            doc_path = os.path.join(root, file)
            doc = codecs.open(doc_path, 'r', 'utf-8').read()
            corpus.append(doc)
    print("Total documents: ", len(corpus))

    random.shuffle(corpus)
    p = int(len(corpus)*0.9)
    train = corpus[:p]
    test = corpus[p:]

    with codecs.open(lda_model_dir+"/"+"corpus_train.dat", 'w', 'utf-8') as datfile:
        datfile.write(str(len(train)) + '\n')
        for doc in train:
            datfile.write(doc + '\n')
    with codecs.open(lda_model_dir+"/"+"corpus_test.dat", 'w', 'utf-8') as testfile:
        testfile.write(str(len(test)) + '\n')
        for doc in test:
            testfile.write(doc + '\n')
    print("Train documents: ", len(train))
    print("Test documents: ", len(test))


if __name__ == '__main__':
    print("Test functions in 'preprocessing.py'.")
    #clean_docs()
    #clean_chars()
    #participle()
    #remove_lf_words()
    #divide_corpus()