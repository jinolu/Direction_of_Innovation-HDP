import pandas as pd
import itertools
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


################################
# Read Data and Transform Data #
################################
def topiccol(df):
    if pd.isnull(df["value"]):
        return df["word"]
    else:
        return np.nan


# Read all files and save in a nested dictionary format in order to calcate similarity.
# Year|Topic as the key for the outer dictionary
file = "./hdp_data_app/3_doc_topics_words/topics.csv"
topic = pd.read_csv(file, header=None, names=["word", "value"])
topic["topic"] = topic.apply(topiccol, axis=1)
topic["topic"] = topic["topic"].fillna(method="ffill")
topic = topic.dropna(subset=["value"])
topic["word"] = topic["word"].fillna("na_placeholder")
topic["topic"] = topic["topic"].str.extract("(\d+)", expand=False)
topic["topic"] = topic["topic"].astype("int")
topic = topic.sort_values(by=["topic", "value"], ascending=[True, False]).reset_index(drop=True)

topic_dict = topic.groupby("topic")[["word", "value"]].apply(lambda x: dict(zip(x["word"], x["value"]))).to_dict()

# Build combination of pairs to calculate similarity later
topic_list = sorted(topic_dict.keys())
pair_topic = list(itertools.combinations(topic_list, 2))  # combination for pairs of Topic


########################
# Calculate Similarity #
########################
similarity_matrix = []
for pair in tqdm(pair_topic):
    # pair=pair_topic[0]
    vec = DictVectorizer()  # Transform the dictionary to Vectorizer in order to measure similarity
    topic1 = pair[0]
    topic2 = pair[1]
    vec_pair = [topic_dict[topic1], topic_dict[topic2]]
    feature_matrix = vec.fit_transform(vec_pair).toarray()
    similarity = cosine_similarity(feature_matrix)[0][1]
    similarity_matrix.append([topic1, topic2, similarity])

similarity_panel = pd.DataFrame(columns=["ParentTopic", "ComparedToTopic",
                                         "IndexofSimilarity"], data=similarity_matrix)

similarity_panel = similarity_panel.sort_values(["ParentTopic", "ComparedToTopic"])
similarity_panel.to_csv(
    "hdp_data_app/3_doc_topics_words/topic_similarity.csv", index=False)
