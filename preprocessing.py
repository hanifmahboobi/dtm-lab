# -*- coding:utf-8 -*-
import codecs
import os
import re
import csv
import jieba
import random
import matplotlib.pyplot as plt


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
                out_file = str(d_info.date[:-1].strip())+'_'+str(d_info.publish[:-1].strip())+\
                           '_'+str(d_info.title.strip()[:2])+'.txt'
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


def is_instr(str):
    if "一" in str or \
                    "二" in str or \
                    "三" in str or \
                    "四" in str or \
                    "五" in str or \
                    "六" in str or \
                    "七" in str or \
                    "八" in str or \
                    "九" in str or \
                    "十" in str:
        return True
    else:
        return False


# Stage 3: Participle and remove stop words and merge synonyms
def participle():

    stopwords_path = "./setting/stop_words.txt"
    synonyms_path = "./setting/synonyms_words.txt"
    user_dicts = "./setting/user_defined_dicts.txt"
    input_dir = "./cleaned_data/stage_2_out"
    output_dir = "./cleaned_data/stage_3_out"

    # load user-defined dicts
    jieba.load_userdict(user_dicts)

    # load synonyms
    combine_dict = {}
    for line in codecs.open(synonyms_path, 'r', 'utf-8'):
        seperate_word = line.strip().split('-')
        num = len(seperate_word)
        for i in range(1, num):
            combine_dict[seperate_word[i]] = seperate_word[0]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # remove stop words and merge synonyms
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
                if (not (word.strip() in stop_seg_list)) and \
                        (not is_instr(word.strip())) and len(word.strip()) > 1:
                    if word.strip() in combine_dict:
                        word_list.append(combine_dict[word])
                    else:
                        word_list.append(word)
            word_list = " ".join(word_list)
            fout.write(word_list)
    print("Stage 3 finished.")


# Stage 4: Remove low-frequency and high-frequency words in global corpus
def remove_lh_words():

    lh_words_path = "./setting/model_params.txt"
    input_dir = "./cleaned_data/stage_3_out"
    output_dir = "./cleaned_data/stage_4_out"
    out_dict_file = "./cleaned_data/temp_dicts.csv"
    word_set = set()
    freq_dict = dict()
    doc_num = 0

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # record all unique words
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            doc_path = os.path.join(root, file)
            doc_num += 1
            with codecs.open(doc_path, 'r', 'utf-8') as doc:
                for line in doc:
                    for word in line.split():
                        if word not in word_set:
                            word_set.add(word)
                            freq_dict[word] = 0
                        else:
                            freq_dict[word] += 1

    lh_setting = codecs.open(lh_words_path, 'r', 'utf-8')
    lf_threshold = 0
    hf_threshold = 1
    for line in lh_setting:
        if line.startswith("low_frequency_threshold"):
            lf_threshold = float(line.split('=')[1])
        elif line.startswith("high_frequency_threshold"):
            hf_threshold = float(line.split('=')[1])
    print("Removing low-frequency words <", lf_threshold)
    print("Removing high-frequency words >", hf_threshold)

    for (k, v) in freq_dict.items():
        if v < lf_threshold*doc_num:
            word_set.remove(k)
        freq_dict[k] = 0

    print("Total unique words: ", len(word_set))
    print("Total documents: ", doc_num)

    for word in word_set:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                doc_path = os.path.join(root, file)
                doc = codecs.open(doc_path, 'r', 'utf-8').read()
                if word in doc:
                    freq_dict[word] += 1

    print("==================================")
    '''
    csvfile = file(out_dict_file, 'w')
    writer = csv.writer(csvfile)
    for word in word_set:
        print(word, "\t", freq_dict[word])
        writer.writerow([str(word), str(freq_dict[word])])
    csvfile.close()
    '''
    with open(out_dict_file, 'w') as out_dict:
        writer = csv.writer(out_dict)
        for word in word_set:
            print(word, "\t", freq_dict[word])
            writer.writerow([str(word), str(freq_dict[word])])

    # remove low-frequency and high-frequency words
    lh_words = set()

    for (k, v) in freq_dict.items():
        if v < lf_threshold*doc_num or v > hf_threshold*doc_num:
            lh_words.add(k)
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            doc_path = os.path.join(root, file)
            out_file = output_dir+'/'+file
            fin = codecs.open(doc_path, 'r', 'utf-8')
            fout = codecs.open(out_file, 'w', 'utf-8')
            word_list = fin.read()
            store_words = []
            for word in word_list.split(" "):
                if not (word.strip() in lh_words) and len(word.strip()) > 1:
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


# Plot how many samples are computed
def plot_samples():
    directory = "./cleaned_data/stage_4_out"
    figure_dir = "./Figures"
    years = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            year = int(f.split('.')[0])
            years.append(year)
    years.sort()
    uniq_years = set(years)
    uniq_years = list(uniq_years)
    uniq_years.sort()

    time_window = {}
    for year in years:
        time_window[year] = years.count(year)

    timestaps = list(time_window.keys())
    num_timestap = list(time_window.values())
    plt.figure()
    plt.bar(timestaps, num_timestap, 0.8, color="blue")
    plt.title('Samples Description')
    plt.xlabel('Year')
    plt.ylabel('Number')

    plt.savefig(figure_dir + '/' + 'samples-description.png')
    plt.show()


if __name__ == '__main__':
    print("Test functions in 'preprocessing.py'.")
    #clean_docs()
    #clean_chars()
    #participle()
    #remove_lh_words()
    #divide_corpus()
    #plot_samples()