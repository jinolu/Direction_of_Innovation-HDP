import sys
import numpy as np


def find_doc_topic_distribution(doc_states, paper_id_file, output_file, p):
    doc_topic_values = []
    with open(doc_states, "r") as f:
        for line in f.readlines():
            line_values = [int(value) for value in line.split(" ")]
            total = sum(line_values)
            if total > 0:
                doc_topic_values.append([(i+1, float(line_values[i])/total)
                                         for i in range(len(line_values)) if float(line_values[i])/total >= p])

    #paper_ids = np.loadtxt(paper_id_file, dtype=int)
    with open(paper_id_file, "r") as f:
        paper_ids = [x.strip() for x in f.readlines()[1:]]
    num_papers = len(paper_ids)
    num_docs = len(doc_topic_values)
    num_topics = len(doc_topic_values[0])
    print("PapereID size: " + str(num_papers))
    print("Prob file size: " + str(num_docs) + " x " + str(num_topics))
    # print "Firsts and lasts: ",paper_ids[0],paper_ids[num_papers-1],
    # print doc_topic_values[0][0],doc_topic_values[0][num_topics-1],doc_topic_values[num_docs-1][0],doc_topic_values[num_docs-1][num_topics-1]
    # raw_input()
    for doc_i in range(len(doc_topic_values)):
        doc_topic_values[doc_i] = sorted(doc_topic_values[doc_i], reverse=True, key=lambda x: x[1])
    with open(output_file, "w") as f_out:
        f_out.write("Document ID,Topic No,Probability\n")
        for doc_i in range(len(doc_topic_values)):
            for (topic_ind, topic_val) in doc_topic_values[doc_i]:
                f_out.write(str(paper_ids[doc_i])+","+str(topic_ind)+","+str(topic_val)+"\n")


doc_states = f"hdp_data_app/2_HDP_results/iter@01000.doc.states"
paper_id_file = f"hdp_data_app/1_paper_ids/all_years_docID.txt"
output_file = f"hdp_data_app/3_doc_topics_words/doc_topics.csv"
p = float(0.01)
find_doc_topic_distribution(doc_states, paper_id_file, output_file, p)
