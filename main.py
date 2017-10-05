#!/usr/bin/python
# -*- coding: utf-8 -*-
import text2ldac
import preprocessing as pre
import lda
import dtm


'''
Followings are pre-processing steps
'''
# Stage 1
pre.clean_docs()
# Stage 2
pre.clean_chars()
# Stage 3
pre.participle()
# Stage 4
pre.remove_lf_words()
# Generate train/test data
pre.divide_corpus()

'''
Using text2ldac to convert documents into the file format used by David Blei's lda-c,
it generates the .dat, .vocab and .dmap files from .txt files in a given directory.
'''
text2ldac.gen_ldac_corpus()

'''
Using 'GibbsLDA++' to run LDA topic model for selecting appropriate number of topics
'''
# Run LDA model with Gibbs sampling
lda.lda_estimate()
# Using trained model to do inference on test set
lda.lda_inference()
# Plot "perplexity" to "number of topics" of LDA according to model results
lda.plot_perplexity()

'''
Run Dynamic Topic Model(DTM) and visualize topics vary with time
'''
# Run Dynamic Topic Model(DTM)
dtm.dtm_estimate()
# Visualize word-time from DTM output
dtm.show_word_times()
# Visualize topic-time from DTM output
dtm.cal_topic_times()
# Visualize standard deviation of topics
dtm.cal_stdvar()
# Visualize structual-changes of topic to times
dtm.cal_strucchange()