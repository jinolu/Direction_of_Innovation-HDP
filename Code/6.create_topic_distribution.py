import sys
import numpy as np
import pandas as pd
from collections import defaultdict


def print_topics(word_count_file, vocab_file, topics_filename, top_n):
    word_counts = []
    with open(word_count_file, "r") as f:
        for line in f.readlines():
            line_values = [int(value) for value in line.split(" ")]
            total = sum(line_values)
            word_counts.append([(i, float(line_values[i])/total) for i in range(len(line_values))])
    # np.loadtxt(word_count_file)
    #vocab = np.loadtxt(vocab_file, dtype=str)
    #num_vocab = len(vocab)
    # print num_vocab
    # print vocab[0],vocab[num_vocab-1]
    num_topics = len(word_counts)
    with open(vocab_file, "r") as f:
        vocab = [x.strip() for x in f.readlines()]
    num_vocab = len(vocab)
    print("Vocab size: " + str(num_vocab))
    print("Prob file size: " + str(num_topics) + " x " + str(len(word_counts[0])))
    # raw_input()
    # print "Firsts and lasts: ",vocab[0],vocab[num_vocab-1],word_counts[0][0],word_counts[0][len(word_counts[0])-1],
    # print word_counts[num_topics-1][0],word_counts[num_topics-1][len(word_counts[0])-1]
    # raw_input()
    for k in range(num_topics):
        word_counts[k] = sorted(word_counts[k], reverse=True, key=lambda x: x[1])[:int(top_n)]

    #prob = np.zeros_like(word_counts)
    '''
    topic_words_map = defaultdict(list)
    print word_counts.shape
    for k in range(num_topics):
        word_prob = word_counts[k,:]/np.sum(word_counts[k,:])
        top_word_idx = word_prob.argsort()[::-1][:int(top_n)]
        word_prob = word_prob[top_word_idx]
        for ind in top_word_idx:
            top_words.append(vocab[ind])
        print word_prob
        print top_word_idx
        print top_words
        topic_words_map[k] = zip(top_words,word_prob)
    '''
    with open(topics_filename, "w") as f_out:
        for k in range(num_topics):
            f_out.write("TOPIC-"+str(k+1)+"\n")
            for i, p in word_counts[k]:
                # for w,p in topic_words_map[k]:
                f_out.write(str(vocab[i])+","+str(p)+"\n")


# USAGE: python print_topics.py final.topics vocab-file output-file num_words

word_count_file = f"hdp_data_app/2_HDP_results/iter@01000.topics"
vocab_file = f"hdp_data_app/1_preprocessed_vector/vocaball_years.txt"
topics_filename = f"hdp_data_app/3_doc_topics_words/topics.csv"
print_topics(word_count_file, vocab_file, topics_filename, 30)
