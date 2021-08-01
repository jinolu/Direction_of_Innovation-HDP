**Replication**

This folder contains the code to apply the algorithm to 44 years of USPTO utility patents between 1976 and 2019:

1.clean_raw_patent.py: Clean raw data (drop duplicates, drop patents with less than five words in abstract); here we use the patent data from PatentsView as input for the application exercise (https://patentsview.org/download/data-download-tables)

2.create_hdp_input.py: (1) Create mappings of document IDs (document IDs required by the HDP to patent IDs) (2) Format the text: lowercasing, tokenization, removal of stop words, of words with single character or numbers only, and of punctuation characters, and lemmatization (3) Build bigrams (4) Create the bag-of-words format with word IDs to words mappings

3.hdp-faster: Run the HDP algorithm and generate two sets of outputs. To run this, please redirect to the 3.hdp-faster folder and run the following code:
./hdp --verbose --train_data /**Your Directory**/Direction_of_Innovation-HDP/Code/1_preprocessed_vector/all_years.dat --directory /**Your Directory**/Direction_of_Innovation-HDP/Code/2_HDP_results --max_time -1 --max_iter 1000 --save_lag 50


4.HDP_iteration_plot.py: Plot the likelihood metrics and algorithm-identified number of topics per number of iterations

5.create_doc_distribution.py: Replace document IDs with patent IDs; filter the sets of topics describing each patent with probability greater than 0.01; get patent-topic distribution

6.create_topic_distribution.py: Replace word IDs with actual words; filter the top 30 words describing each topic with corresponding probabilities; get topic-word distribution

7.compute_topic_similarity.py: Compute pairwise weighted cosine similarity between the HDP topics

8.HHI_diversification.py: (1) Match patents with patent assignees (or inventors, industries, or regions) (2) Compute HDP- and CPC-based HHI diversification measures for benchmarking
