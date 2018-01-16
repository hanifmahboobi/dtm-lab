#!/usr/bin/python
# -*- coding: utf-8 -*-
import codecs
import text2ldac
import text2code
import preprocessing as pre
import lda
import dtm


start_path = "./说明.txt"
step = 0
with codecs.open(start_path, 'r', 'utf-8') as pfile:
    for line in pfile:
        if line.startswith("start"):
            step = int(line.strip().split('=')[1])
print(step)
if step == 1:
    # Stage 1
    pre.clean_docs()
elif step == 2:
    # Stage 2
    pre.clean_chars()
elif step == 3:
    # Stage 3
    pre.participle()
elif step == 4:
    # Stage 4
    pre.remove_lh_words()
    # Generate train/test data
    pre.divide_corpus()
elif step == 5:
    # Plot how many samples are computed
    pre.plot_samples()
elif step == 6:
    text2ldac.gen_ldac_corpus()
    # Run LDA model with Gibbs sampling
    ntopics = lda.lda_estimate()
    # Using trained model to do inference on test set
    lda.lda_inference(ntopics)
    # Plot "perplexity" to "number of topics" of LDA according to model results
    lda.plot_perplexity(ntopics)
elif step == 7:
    # Run Dynamic Topic Model(DTM)
    dtm.dtm_estimate()
elif step == 8:
    # Visualize word-time from DTM output
    dtm.show_word_times()
elif step == 9:
    # Visualize topic-doc from DTM output
    dtm.show_topic_docs()
elif step == 10:
    # Visualize topic-time from DTM output
    dtm.cal_topic_times()
elif step == 11:
    # Visualize structual-changes of topic to times
    dtm.cal_strucchange()
