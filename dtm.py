# -*- coding:utf-8 -*-
import os
import codecs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def set_time_window(time_interval):

    directory = "./cleaned_data/stage_4_out"
    out_filename = "./db/cleaned_data-seq.dat"

    years = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            year = int(f.split('.')[0])
            years.append(year)
    time_window = {}
    if time_interval == 1:
        for year in years:
            time_window[year] = years.count(year)
    elif time_interval >= 2:
        # To be extended next time
        print("To be extended next time...")
    else:
        pass

    number_timestaps = len(time_window)
    # timestaps = time_window.keys()
    num_timestap = time_window.values()

    with codecs.open(out_filename, 'w', 'utf-8') as out_file:
        out_file.write(str(number_timestaps) + '\n')
        for num in num_timestap:
            out_file.writelines(str(num) + '\n')


# Run Dynamic Topic Model(DTM)
def dtm_estimate():

    corpus_prefix = "./db/cleaned_data"
    out_name = "./models/dtm"
    param_path = "./setting/dtm_params.txt"

    # get parameters from setting file
    with codecs.open(param_path, 'r', 'utf-8') as pfile:
        for line in pfile:
            if line.startswith("ntopics"):
                ntopics = line.strip().split('=')[1]
            if line.startswith("time"):
                time_intval = int(line.split('=')[1])
    # generate time slice file
    set_time_window(time_intval)

    # call external C++ exe to run
    os.chdir("./lib/DTM/bin")
    params = "--ntopics="+str(ntopics)+" --mode=fit --rng_seed=0 --initialize_lda=true" \
             " --corpus_prefix=../../../"+str(corpus_prefix)+" --outname=../../../"+str(out_name)+\
             " --top_chain_var=0.005 --alpha=0.01 --lda_sequence_min_iter=6" \
             " --lda_sequence_max_iter=20 --lda_max_em_iter=6"
    os.system("dtm-win32.exe "+params)


# Calculate topic-time matrix from DTM output
def cal_topic_times():

    timeslice_file_path = "./db/cleaned_data-seq.dat"
    gam_file_path = "./models/dtm/lda-seq/gam.dat"
    param_path = "./setting/dtm_params.txt"
    out_file = "./db/topic_times.csv"
    figure_dir = "./Figures"

    # get parameters from setting file
    topics = []
    with codecs.open(param_path, 'r', 'utf-8') as pfile:
        for line in pfile:
            if line.startswith("ntopics"):
                num_topics = int(line.strip().split('=')[1])
            elif line.startswith("topic"):
                topics.append(line.strip().split('=')[1])

    gammas = pd.read_table(gam_file_path, header=None)
    gammas = np.array(gammas)
    gammas = gammas.reshape((-1, num_topics))
    gammas = pd.DataFrame(gammas)
    time_slice = pd.read_table(timeslice_file_path)
    time_slice = np.asarray(time_slice)
    num_time_slice = len(time_slice)

    # store topic-time matrix
    gam_sum = np.zeros((num_time_slice, num_topics))
    row_i = 0
    for i in range(num_time_slice):
        gam_year = gammas.iloc[row_i:row_i+time_slice[i][0]]
        gam_sum[i, :] = gam_year.apply(lambda x: x.sum(), axis=0)
        row_i = row_i + time_slice[i][0]
    gam_sum = pd.DataFrame(gam_sum)
    gam_sum['sum'] = gam_sum.apply(lambda x: x.sum(), axis=1)
    for i in range(num_time_slice):
        sum_value = gam_sum.ix[i][-1]
        gam_sum.ix[i, :-1] = gam_sum.ix[i, :-1].apply(lambda x: x/sum_value)
    result = gam_sum.ix[:, :-1]
    result.to_csv(out_file, header=False, index=False)

    # visualizing topics-times
    colors = ["magenta", "yellow", "black", "red", "orange", "purple", "green", "cyan", "blue", "gray"]
    date_list = list(range(1950, 2018))
    date_list.insert(0, 1948)
    font = FontProperties(fname='C:\Windows\Fonts\msyh.ttc')
    fig, ax = plt.subplots()
    for i in range(num_topics):
        legend_name = topics[i]
        ax.plot(date_list, result.ix[:, i], 'k--', label=legend_name, color=colors[i])
    ax.legend(prop=font)
    plt.ylabel("Relative weight")
    plt.xlabel("Year")

    plt.savefig(figure_dir + '/' + 'topic-time.png')
    plt.show()


# Calculate word-time matrix from DTM output
def cal_word_times(topic_no, time_slice, k_term=8):

    topic_file_path = "./models/dtm/lda-seq/"
    topic_file_name = "topic-00"+str(topic_no)+"-var-e-log-prob.dat"
    vocab_file_path = "./db/cleaned_data.vocab"
    outfile_prefix = "./db/word-times_topic" + str(topic_no)
    figure_dir = "./Figures"

    matrix = pd.read_table(topic_file_path+topic_file_name, header=None)
    matrix = np.array(matrix)
    matrix = matrix.reshape((-1, time_slice))
    matrix = np.exp(matrix)
    matrix = pd.DataFrame(matrix)
    # print "matrix:\n", matrix
    vocab = pd.read_table(vocab_file_path, header=None, encoding='utf-8')
    # print "vocab:\n", vocab
    # count total prob. of each term in all time slices
    matrix['sum'] = matrix.apply(lambda x: x.sum(), axis=1)
    top_k_term = sorted(np.array(matrix['sum']), reverse=True)
    y_vars = []
    var_names = []

    for i in range(k_term):
        top_k = top_k_term[i]
        index = np.where(matrix['sum'] == top_k)
        index = int(index[0])
        y_vars.append(matrix.ix[index, :-1])
        var_names.append(vocab.ix[index][0])
        print(vocab.ix[index][0])

    # visualizing terms-times in topic "topic_no"
    colors = ["blue", "red", "black", "orange", "green", "cyan", "gray", "purple"]
    font = FontProperties(fname='C:\Windows\Fonts\msyh.ttc')
    date_list = list(range(1950, 2018))
    date_list.insert(0, 1948)
    fig, ax = plt.subplots()
    for i in range(k_term):
        legend_name = var_names[i]
        ax.plot(date_list, y_vars[i], 'k--', marker='*', label=legend_name, color=colors[i], linewidth=2)
    ax.legend(prop=font, )
    plt.title("Topic"+str(topic_no))
    plt.ylabel("Probability.")
    plt.xlabel("Year")

    plt.savefig(figure_dir + '/' + 'word-time_topic'+str(topic_no)+'.png')
    plt.show()

    y_vars = pd.DataFrame(y_vars)
    y_vars.to_csv(outfile_prefix + '.csv', header=False, index=False)


def show_word_times():

    param_path = "./setting/dtm_params.txt"
    timeslice_file_path = "./db/cleaned_data-seq.dat"

    time_slice = pd.read_table(timeslice_file_path)
    time_slice = np.asarray(time_slice)
    num_time_slice = len(time_slice)
    with codecs.open(param_path, 'r', 'utf-8') as pfile:
        for line in pfile:
            if line.startswith("ntopics"):
                num_topics = int(line.strip().split('=')[1])

    for t in range(num_topics):
        cal_word_times(t, num_time_slice)


# Calculate Standard deviation of each time slice in topic-time matrix
def cal_stdvar():

    file_path = "./db/topic_times.csv"
    figure_dir = "./Figures"

    topic_time = pd.read_csv(file_path, header=None)
    topic_time['var'] = topic_time.apply(lambda x: np.std(x), axis=1)
    date_list = list(range(1950, 2018))
    date_list.insert(0, 1948)
    x = date_list
    y = topic_time['var']
    plt.plot(x, y, marker="+", color="red", linewidth=2)
    plt.xlabel("Year")
    plt.ylabel("Standard deviation")

    plt.savefig(figure_dir + '/' + 'topics.std.png')
    plt.show()


# Plot structual-changes of topic to times
def cal_strucchange():

    os.system("R CMD BATCH --args structual_change.R")


if __name__ == "__main__":
    print("Test functions in 'dtm.py'.")
    #dtm_estimate()
    #show_word_times()
    #cal_topic_times()
    #cal_stdvar()
    #cal_strucchange()