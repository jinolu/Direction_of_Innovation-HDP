
1. clean_raw_patent.py: Clean raw data (drop duplicates, drop patents with less than five words in abstract)
2. create_hdp_input.py: (1) Create mappings of document IDs (document IDs required by the HDP to patent IDs) (2) Format the text: lowercasing, tokenization, removal of stop words, of words with single character or numbers only, and of punctuation characters, and lemmatization (3) Create the bag-of-words format with word IDs to words mappings
3. hdp-faster: Run the HDP algorithm and generate two sets of outputs
4. create_doc_distribution.py: Replace document IDs with patent IDs; filter the sets of topics describing each patent with probability greater than 0.01; get patent-topic distribution
5. create_topic_distribution.py: Replace word IDs with actual words; filter the top 30 words describing each topic with corresponding probabilities; get topic-word distribution
6. compute_topic_similarity.py: Compute pairwise weighted cosine similarity between the yearly HDP topics
7. research_diversification.py: (1) Match patents with patent assignees (or inventors, industries, or regions) (2) Compute diversification measures (intensive and extensive) and trajectory measure
8. research_diversification_alternative_measure.py: Compute patent class-based diversification and trajectory measure
