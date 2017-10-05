# -*- coding:utf-8 -*-
import os
import codecs
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Run LDA model with Gibbs sampling
def lda_estimate():

    train_path = "./models/lda/corpus_train.dat"
    test_path = "./models/lda/corpus_test.dat"
    params_path = "./setting/lda_params.txt"

    # get parameters from setting file
    with codecs.open(params_path, 'r', 'utf-8') as pfile:
        for line in pfile:
            if line.startswith("alpha"):
                alpha = line.split('=')[1].strip()
            elif line.startswith("beta"):
                beta = line.split('=')[1].strip()
            elif line.startswith('ntopics'):
                line = line.split('=')[1]
                line = line[1:-3].split(',')
                ntopics = list(int(line[i]) for i in range(len(line)))
            elif line.startswith('niters'):
                niters = line.split('=')[1].strip()
            elif line.startswith('savestep'):
                savestep = line.split('=')[1].strip()
            elif line.startswith('twords'):
                twords = line.split('=')[1].strip()
    # create dirs for LDA model output
    for topic in ntopics:
        tdir = "./models/lda/topic_"+str(topic)
        if not os.path.exists(tdir):
            os.mkdir(tdir)
        tfile = tdir + "/corpus_train.dat"
        testfile = tdir + "/corpus_test.dat"
        if not os.path.exists(tfile):
            shutil.copyfile(train_path, tfile)
        if not os.path.exists(testfile):
            shutil.copyfile(test_path, testfile)

    # call external C++ exe to run LDA topic model
    os.chdir("./lib/GibbsLDA++/bin")
    for topic in ntopics:
        print("Training LDA model with", topic, "topics...")
        dfile = "../../../models/lda/topic_"+str(topic)+"/corpus_train.dat"
        params = "-alpha "+str(alpha)+" -beta "+str(beta)+" -ntopics "+str(topic)\
                 + " -niters "+str(niters)+" -savestep "+str(savestep)+" -twords "+str(twords)\
                 + " -treval 1"+" -dfile "+str(dfile)
        os.system("lda.exe -est "+params)
        print("Training finished.")
    print("Finished.")


# Using trained model to do inference on test set
def lda_inference():

    params_path = "./setting/lda_params.txt"
    with codecs.open(params_path, 'r', 'utf-8') as pfile:
        for line in pfile:
            if line.startswith('ntopics'):
                line = line.split('=')[1]
                line = line[1:-3].split(',')
                ntopics = list(int(line[i]) for i in range(len(line)))
    # call external C++ exe to run LDA inference
    os.chdir("./lib/GibbsLDA++/bin")
    dir_prefix = "../../../models/lda/topic_"
    for topic in ntopics:
        print("Inference on test set with", topic, "topics...")
        tfile = dir_prefix+str(topic)+"/corpus_test.dat"
        dir = dir_prefix+str(topic)+'/'
        params = "-dir "+dir+" -model model-final -niters 20 -twords 50 -treval 1 -teval 1 -dfile "+tfile
        os.system("lda.exe -inf " + params)
        print("Inference finished.")


def figure_plot(topic, perplexity):
    x = topic
    y = perplexity
    plt.plot(x, y, marker="*", color="red", linewidth=2)
    plt.xlabel("Number of Topics")
    plt.ylabel("Perplexity")

    figure_dir = "./Figures"
    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)

    plt.savefig(figure_dir + '/' + 'perplexity.png')
    plt.show()


# Calculate perplexity of LDA
def cal_perplex(topic_word, doc_topic, tassign):

    topic_word = np.mat(np.asarray(topic_word))
    doc_topic = np.mat(np.asarray(doc_topic))
    doc_word = doc_topic * topic_word
    doc_word = pd.DataFrame(doc_word)
    print(doc_word.shape)

    log_pw = doc_word.apply(lambda x: np.log(x.sum()), axis=0)
    sum_log_pw = np.sum(log_pw)
    print("sum_log_pw:", sum_log_pw)

    sum_t = 0
    for i in range(len(tassign)):
        doc_i = tassign[i].strip().split(" ")
        sum_t += len(doc_i)
    print("sum_t:", sum_t)

    '''
    for i in range(len(tassign)):
        doc_i = tassign[i].strip().split(" ")
        print(len(doc_i))
        sum_t += len(doc_i)
        log_pw = 0.0
        for word in doc_i:
            pz = 0.0
            word_id = int(word.split(':')[0])
            for j in range(len(topic_word)):
                if word_id in range(topic_word.shape[1]):
                    p_tz = float(topic_word.ix[j][word_id])
                else:
                    p_tz = 0.0
                    #print("word id:", word_id)
                p_zd = float(doc_topic.ix[i][j])
                pz += p_tz*p_zd
            log_pw += np.log(pz+0.0001)
        sum_log_pw += log_pw
    '''

    perplexity = np.exp(-sum_log_pw/sum_t)
    print("perplex: ", perplexity)
    return perplexity


# Plot "perplexity" to "number of topics" of LDA according to model results
def plot_perplexity():

    dir_prefix = "./models/lda/topic_"
    params_path = "./setting/lda_params.txt"
    with codecs.open(params_path, 'r', 'utf-8') as pfile:
        for line in pfile:
            if line.startswith('ntopics'):
                line = line.split('=')[1]
                line = line[1:-3].split(',')
                ntopics = list(int(line[i]) for i in range(len(line)))
    perplexity_list = []
    for topic in ntopics:
        print("topic", topic)
        dir = dir_prefix + str(topic) + '/'
        px_path = dir+"corpus_test.dat.perplex.txt"
        px_list = codecs.open(px_path, 'r', 'utf-8').read()
        px_list = px_list.strip().split('\n')
        px = float(px_list[-1].strip())
        perplexity_list.append(px)
        print("px:", px)

    figure_plot(ntopics, perplexity_list)


if __name__ == "__main__":
    print("Test functions in 'lda.py'.")
    #lda_estimate()
    #lda_inference()
    #plot_perplexity()