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
year_word_value = {}
for i in tqdm(range(2000, 2019)):
    year = str(i)
    file = "./hdp_data/doc_topics_words/"+year+"/topics_"+year+".csv"
    topic = pd.read_csv(file, header=None, names=['word', "value"])
    topic["topic"] = topic.apply(topiccol, axis=1)
    topic["topic"] = topic["topic"].fillna(method="ffill")
    topic = topic.dropna(subset=["value"])
    topic["word"] = topic["word"].fillna("na_placeholder")
    topic["topic"] = year+"|"+topic["topic"]
    year_dict = topic.groupby("topic")[["word", "value"]].apply(lambda x: dict(zip(x["word"], x["value"]))).to_dict()
    year_word_value.update(year_dict)

pickle.dump(year_word_value, open("year_word_value.p", 'wb'))
#year_word_value=pickle.load(open('year_word_value.p', 'rb'))


# Build combination of pairs to calculate similarity later
year_topic = sorted(year_word_value.keys())  # all Year|Topic
pair_topic = list(itertools.combinations(year_topic, 2))  # combination for pairs of Year|Topic

########################
# Calculate Similarity #
########################
similarity_matrix = []
for pair in tqdm(pair_topic):
    vec = DictVectorizer()  # Transform the dictionary to Vectorizer in order to measure similarity
    yeartopic1 = pair[0]
    yeartopic2 = pair[1]
    vec_pair = [year_word_value[yeartopic1], year_word_value[yeartopic2]]
    feature_matrix = vec.fit_transform(vec_pair).toarray()
    similarity = cosine_similarity(feature_matrix)[0][1]
    year1, topic1 = yeartopic1.split('|')
    year2, topic2 = yeartopic2.split('|')
    similarity_matrix.append([topic1, year1, topic2, year2, similarity])


pickle.dump(similarity_matrix, open("topic_similarity_matrix.p", 'wb'))
similarity_panel = pd.DataFrame(columns=['ParentTopic', 'ParentTopicYear', 'ComparedToTopic', 'ComparedToTopicYear',
                                         'IndexofSimilarity'], data=similarity_matrix)

similarity_panel["parentindex"] = similarity_panel["ParentTopic"].str.extract('(\d+)', expand=False).astype(int)
similarity_panel["comindex"] = similarity_panel["ComparedToTopic"].str.extract('(\d+)', expand=False).astype(int)
similarity_panel = similarity_panel.sort_values(["ParentTopicYear", "ComparedToTopicYear", "parentindex", "comindex"])
similarity_panel = similarity_panel.drop(columns=["parentindex", "comindex"])
similarity_panel.to_csv("topic_similarity_panel.csv", index=False)
