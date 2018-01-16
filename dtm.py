# -*- coding:utf-8 -*-
import os
import codecs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def set_time_window(time_interval):

    directory = "./cleaned_data/stage_4_out"
    out_filename = "./models/db/cleaned_data-seq.dat"
    time_out_filename = "./models/db/time-seq.txt"
    years = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            year = int(f.split('.')[0])
            years.append(year)
    years.sort()
    uniq_years = set(years)
    uniq_years = list(uniq_years)
    uniq_years.sort()

    with codecs.open(time_out_filename, 'w', 'utf-8') as tfile:
        for y in uniq_years:
            tfile.write(str(y) + '\n')

    time_window = {}
    if time_interval == 1:
        for year in years:
            time_window[year] = years.count(year)
    elif time_interval >= 2:
        start = years[0]
        end = years[-1]
        tstap = 1
        current = start
        if end-start < time_interval:
            time_window[tstap] = len(years)
        else:
            while current <= end:
                time_window[tstap] = 0
                for i in range(time_interval):
                    time_window[tstap] += years.count(current)
                    current += 1
                tstap += 1

    number_timestaps = len(time_window)
    # timestaps = time_window.keys()
    num_timestap = time_window.values()

    with codecs.open(out_filename, 'w', 'utf-8') as out_file:
        out_file.write(str(number_timestaps) + '\n')
        for num in num_timestap:
            out_file.writelines(str(num) + '\n')


# Run Dynamic Topic Model(DTM)
def dtm_estimate():

    corpus_prefix = "./models/db/cleaned_data"
    out_name = "./models/dtm"
    param_path = "./setting/model_params.txt"

    # get parameters from setting file
    with codecs.open(param_path, 'r', 'utf-8') as pfile:
        for line in pfile:
            if line.startswith("topics"):
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


# Calculate word-time matrix from DTM output
def cal_word_times(topic_no, time_slice, k_term=8):

    topic_file_path = "./models/dtm/lda-seq/"
    time_file_path = "./models/db/time-seq.txt"
    if topic_no < 10:
        topic_file_name = "topic-00" + str(topic_no) + "-var-e-log-prob.dat"
    else:
        topic_file_name = "topic-0"+str(topic_no)+"-var-e-log-prob.dat"
    vocab_file_path = "./models/db/cleaned_data.vocab"
    outfile_prefix = "./models/db/word-times_topic" + str(topic_no)
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
        var = list(matrix.ix[index, :-1])
        var.insert(0, vocab.ix[index][0])
        y_vars.append(var)
        var_names.append(vocab.ix[index][0])
        print(vocab.ix[index][0])

    # visualizing terms-times in topic "topic_no"
    date_list = []
    with codecs.open(time_file_path, 'r', 'utf-8') as tfile:
        for line in tfile:
            date_list.append(int(line.strip()))
    if k_term <= 20:
        fig, ax = plt.subplots(figsize=(12, 6))
        font = FontProperties(fname='C:\Windows\Fonts\msyh.ttc', size=10)
    elif k_term > 20:
        fig, ax = plt.subplots(figsize=(12, 8))
        font = FontProperties(fname='C:\Windows\Fonts\msyh.ttc', size=8)
    plt.subplots_adjust(left=0.1)
    colors = ["red", "blue", "black", "orange", "purple", "green", "magenta", "cyan", "yellow", "gray"]
    markers = ['+', '.', '*']
    color_index = 0
    marker_index = 0
    for i in range(k_term):
        legend_name = var_names[i]
        ax.plot(date_list, y_vars[i][1:], 'k--', marker=markers[marker_index],
                label=legend_name, color=colors[color_index], linewidth=1)
        color_index += 1
        color_index = color_index % 10
        if color_index == 0:
            marker_index = (marker_index + 1) % 3
    if k_term <= 10:
        ncol = 2
    elif k_term <= 15:
        ncol = 3
    elif k_term <= 20:
        ncol = 4
    elif k_term <= 25:
        ncol = 5
    else:
        ncol = 6
    ax.legend(prop=font, loc='best', ncol=ncol)
    plt.title("Topic"+str(topic_no))
    plt.ylabel("Probability")
    plt.xlabel("Year")

    plt.savefig(figure_dir + '/' + 'word-time_topic'+str(topic_no)+'.png')
    plt.show()

    # write results to csv
    y_vars = pd.DataFrame(y_vars)
    date_list.insert(0, 'Year')
    y_vars.columns = date_list
    y_vars.to_csv(outfile_prefix + '.csv', header=True, index=False, encoding='GBK')


def show_word_times():

    param_path = "./setting/model_params.txt"
    timeslice_file_path = "./models/db/cleaned_data-seq.dat"

    time_slice = pd.read_table(timeslice_file_path)
    time_slice = np.asarray(time_slice)
    num_time_slice = len(time_slice)
    with codecs.open(param_path, 'r', 'utf-8') as pfile:
        for line in pfile:
            if line.startswith("topics"):
                num_topics = int(line.strip().split('=')[1])
            if line.startswith("words"):
                k_term = int(line.strip().split('=')[1])

    for t in range(num_topics):
        cal_word_times(t, num_time_slice, k_term=k_term)


# Calculate Standard deviation of each time slice in topic-time matrix
def cal_stdvar():

    file_path = "./models/db/topic_times.csv"
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


# Calculate topic-time matrix from DTM output
def cal_topic_times():

    timeslice_file_path = "./models/db/cleaned_data-seq.dat"
    gam_file_path = "./models/dtm/lda-seq/gam.dat"
    param_path = "./setting/model_params.txt"
    out_file = "./models/db/topic_times.csv"
    time_file_path = "./models/db/time-seq.txt"
    figure_dir = "./Figures"

    # get parameters from setting file
    topics = dict()
    with codecs.open(param_path, 'r', 'utf-8') as pfile:
        for line in pfile:
            if line.startswith("topics"):
                num_topics = int(line.strip().split('=')[1])
            elif line.startswith("topic"):
                topics[int(line.strip().split('=')[0][5:])] = line.strip().split('=')[1]

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

    # calculate standard deviation of each time slice in topic-time matrix
    topic_time = result
    topic_time['std'] = topic_time.apply(lambda x: np.std(x), axis=1)
    topic_time.to_csv(out_file, header=False, index=False)

    # visualizing topics-times
    if num_topics < 10:
        fig, left_axis = plt.subplots(figsize=(12, 6))
        font = FontProperties(fname='C:\Windows\Fonts\msyh.ttc', size=10)
    elif num_topics >= 10:
        fig, left_axis = plt.subplots(figsize=(12, 8))
        font = FontProperties(fname='C:\Windows\Fonts\msyh.ttc', size=9)
    right_axis = left_axis.twinx()
    date_list = []
    with codecs.open(time_file_path, 'r', 'utf-8') as tfile:
        for line in tfile:
            date_list.append(int(line.strip()))
    colors = ["red", "blue", "black", "orange", "purple", "green", "magenta", "cyan", "yellow", "gray"]
    markers = ['o', '*',  '.']
    color_index = 0
    marker_index = 0
    for (k, v) in topics.items():
        legend_name = v
        left_axis.scatter(date_list, result.ix[:, k], marker=markers[marker_index],
                          label=legend_name, color=colors[color_index], linewidths=1)
        color_index += 1
        color_index = color_index % 10
        if color_index == 0:
            marker_index = (marker_index+1) % 3
    if num_topics <= 10:
        ncol = 1
    elif num_topics <= 20:
        ncol = 2
    else:
        ncol = 3
    left_axis.legend(prop=font, bbox_to_anchor=(0.95, 0.85), ncol=ncol)
    left_axis.set_ylabel("Relative weight")

    y2 = topic_time['std']
    right_axis.plot(date_list, y2, marker="^", label="std.", color="black", linewidth=1)
    right_axis.legend(prop=font, bbox_to_anchor=(0.95, 0.95))
    right_axis.set_ylabel("Standard deviation")

    left_axis.set_xlabel("Year")
    plt.savefig(figure_dir + '/' + 'topic-time.png')
    plt.show()


# Show topic-documents matrix from DTM output
def show_topic_docs():
    gam_file_path = "./models/dtm/lda-seq/gam.dat"
    dmap_file_path = "./models/db/cleaned_data.dmap"
    param_path = "./setting/model_params.txt"
    out_file = "./models/db/topic_docs.csv"
    figure_dir = "./Figures"

    with codecs.open(param_path, 'r', 'utf-8') as pfile:
        for line in pfile:
            if line.startswith("topics"):
                num_topics = int(line.strip().split('=')[1])
            elif line.startswith("docs"):
                k_term = int(line.strip().split('=')[1])
    doc_name = []
    with codecs.open(dmap_file_path, 'r', 'utf-8') as dfile:
        for line in dfile:
            doc_name.append(line.strip().split('\\')[1])

    gammas = pd.read_table(gam_file_path, header=None)
    gammas = np.array(gammas)
    gammas = gammas.reshape((-1, num_topics))
    results = []
    for t in range(num_topics):
        dw = gammas[:, t]/sum(gammas[:, t])
        sort_dw = sorted(dw, reverse=True)
        d = []
        for i in range(k_term):
            top_k = sort_dw[i]
            index = np.where(dw == top_k)
            index = int(index[0])
            dn = doc_name[index].strip()
            dn += " "+str(top_k)
            d.append(dn)
        results.append(d)
    # write results to csv
    df = pd.DataFrame(results)
    df_T = df.T
    df_T.columns = list(range(num_topics))
    df_T.to_csv(out_file, index=False, header=True, encoding='GBK')
    # visualizing topic-docs
    font = FontProperties(fname='C:\Windows\Fonts\msyh.ttc')
    for t in range(num_topics):
        data = df_T.ix[:, t]
        x_var = list(data.apply(lambda x: float(x.strip().split(" ")[1])))
        x_var.reverse()
        y_var = list(data.apply(lambda x: x.strip().split(" ")[0]))
        y_var.reverse()
        print("x_var\n", x_var)
        print("y_var\n", y_var)
        idx = np.arange(len(x_var))
        plt.figure(figsize=(12, 6))
        plt.subplots_adjust(left=0.2)
        plt.barh(idx, x_var)
        plt.yticks(idx, y_var, fontproperties=font)
        plt.grid(axis='x')
        plt.xlabel("Weight")
        plt.ylabel("Document")
        plt.title("Topic "+str(t))
        plt.savefig(figure_dir + '/' + 'topic'+str(t)+'-docs.png')
        plt.show()


# Plot structual-changes of topic to times
def cal_strucchange():

    os.system("R CMD BATCH --args structual_change.R")


if __name__ == "__main__":
    print("Test functions in 'dtm.py'.")
    #set_time_window(1)
    #dtm_estimate()
    #show_word_times()
    #cal_topic_times()
    #show_topic_docs()
    #cal_strucchange()