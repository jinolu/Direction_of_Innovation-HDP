from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from ggplot import *
from tqdm import tqdm
from plotnine import *
import numpy as np
import math


raw_assignee = pd.read_csv("PatentsView/patent_assignee.tsv", sep="\t")
patent_class = pd.read_csv("PatentsView/cpc_current.tsv", sep="\t")
file = "./hdp_data_app/3_doc_topics_words/doc_topics.csv"
doc_topics = pd.read_csv(file, header=0, names=["patent_id", "topic_id", "probability"])
doc_topics = doc_topics.reset_index(drop=True)
doc_topics["patent_id"].nunique()

doc_topics["patent_id"] = doc_topics["patent_id"].astype("str")
raw_assignee["patent_id"] = raw_assignee["patent_id"].astype("str")

# merge with applications to get years
application = pd.read_csv("PatentsView/application.tsv", sep="\t")
application["year"] = application["date"].str[:4]
application["year"] = application["year"].astype("int")
application = application[["patent_id", "year"]]
application["patent_id"] = application["patent_id"].astype("str")
doc_topics = pd.merge(left=doc_topics, right=application, how="left", on="patent_id")

# merge with raw_assignee
assignee_patent_topic = pd.merge(left=doc_topics, right=raw_assignee, how="left", left_on="patent_id", right_on="patent_id")

assignee_patent_topic = assignee_patent_topic[assignee_patent_topic["assignee_id"].isnull() == False]
assignee_patent_topic = assignee_patent_topic[assignee_patent_topic["assignee_id"].str.contains("per_") == False]
assignee_patent_topic = assignee_patent_topic[["assignee_id", "year", "patent_id", "topic_id", "probability"]]
assignee_patent_topic = assignee_patent_topic.sort_values(
    by=["assignee_id", "year", "patent_id", "topic_id"]).reset_index(drop=True)
assignee_patent_topic = assignee_patent_topic.drop_duplicates()
assignee_patent_topic = assignee_patent_topic.sort_values(by=["assignee_id", "year", "topic_id"])
assignee_patent_topic["year"] = assignee_patent_topic["year"].astype("int")


############################
# Merge with assignee name #
############################
assignee = pd.read_csv(
    "PatentsView/assignee.tsv", sep="\t", usecols=["id", "type", "organization"])
assignee = assignee.drop_duplicates()
assignee["organization"] = assignee["organization"].str.lower()
assignee = assignee[assignee["organization"].notnull()]
assignee = assignee[assignee["type"].isin([2, 3])]
assignee[assignee["organization"].isnull()]
assignee["organization"] = assignee["organization"].str.strip()

assignee = assignee.rename(columns={"id": "assignee_id"})

# Merge firm with assignee_doc_topics
assignee_patent_topic1 = pd.merge(left=assignee, right=assignee_patent_topic, how="left", on="assignee_id")
assignee_patent_topic1 = assignee_patent_topic1[assignee_patent_topic1["year"].notnull()]
assignee_patent_topic1["topic_id"] = assignee_patent_topic1["topic_id"].astype("int")
assignee_patent_topic1["year"] = assignee_patent_topic1["year"].astype("int")
assignee_patent_topic1 = assignee_patent_topic1.sort_values(
    by=["assignee_id", "year", "patent_id", "topic_id"]).reset_index(drop=True)
# assignee_patent_topic1.to_csv(
#    "hdp_data_app/4_cpc_benchmarking/assignee_patent_topic.csv", index=False)


############################################################################
#                    HDP-based HHI diversification measure                 #
############################################################################
print("HDP-based HHI diversification measure")
# assignee_patent_topic1 = pd.read_csv(
#    "hdp_data_app/4_cpc_benchmarking/assignee_patent_topic.csv")
assignee_patent_topic1 = assignee_patent_topic1.drop_duplicates(
    ["assignee_id", "year", "patent_id", "topic_id"]).reset_index(drop=True)

assignee_patent_topic1["n_patent_focal_year"] = assignee_patent_topic1.groupby(["assignee_id", "year"])[
    "patent_id"].transform("nunique")
assignee_year_topic = assignee_patent_topic1.groupby(["assignee_id", "year", "topic_id"])["probability", "n_patent_focal_year"].agg({
    "probability": "sum", "n_patent_focal_year": "first"}).reset_index()

assignee_year_topic["weighted_prob"] = assignee_year_topic["probability"]/assignee_year_topic["n_patent_focal_year"]
assignee_year_topic["weighted_prob_sq"] = assignee_year_topic["weighted_prob"]**2
assignee_hhi_hdp = assignee_year_topic.groupby(["assignee_id", "year"])["weighted_prob_sq"].sum().reset_index()
assignee_hhi_hdp = assignee_hhi_hdp.rename(columns={"weighted_prob_sq": "hhi_hdp"})
assignee_hhi_hdp["hhi_diversity_hdp"] = 1-assignee_hhi_hdp["hhi_hdp"]


###########################################################################
#             Patent Class-based HHI diversification measure              #
###########################################################################
print("Patent Class-based diversification measure")
assignee_patent_topic1 = pd.read_csv("hdp_data_app/4_cpc_benchmarking/assignee_patent_topic.csv")
firm_patent = assignee_patent_topic1[["assignee_id", "year", "patent_id"]].drop_duplicates()

# load patent classes
patent_class = pd.read_csv(
    "PatentsView/cpc_current.tsv", sep="\t", usecols=["patent_id",  "group_id"])
patent_class = patent_class.drop_duplicates().reset_index(drop=True)

# merge assignee_patent with patent class
firm_patent["patent_id"] = firm_patent["patent_id"].astype("str")
patent_class["patent_id"] = patent_class["patent_id"].astype("str")

assignee_cpc = pd.merge(left=firm_patent, right=patent_class, how="left", left_on="patent_id", right_on="patent_id")
assignee_cpc = assignee_cpc[assignee_cpc["group_id"].isnull() == False]

assignee_cpc = assignee_cpc.drop_duplicates().sort_values(by=[
    "assignee_id", "year", "patent_id"]).reset_index(drop=True)

#assignee_cpc.to_csv("hdp_data_app/4_cpc_benchmarking/assignee_name_cpc_patent_class.csv", index=False)


###################################################
# Build patent class HHI diversification measures #
###################################################
print("Build patent class HHI diversification measures")


def hhi(series):
    _, cnt = np.unique(series, return_counts=True)
    return np.square(cnt/cnt.sum()).sum()


assignee_hhi_cpc = assignee_cpc.groupby(["assignee_id", "year"])["patent_id", "group_id"].agg(
    {"patent_id": "nunique", "group_id": hhi}).reset_index()
assignee_hhi_cpc = assignee_hhi_cpc.rename(columns={"patent_id": "n_patent", "group_id": "hhi_cpc"})
assignee_hhi_cpc["hhi_diversity_cpc"] = 1-assignee_hhi_cpc["hhi_cpc"]


##################################################################
# Merge hdp-based and patent class-based diversification measure #
##################################################################
print("Merge hdp-based and patent class-based HHI diversification measure")
assignee_hhi = pd.merge(left=assignee_hhi_hdp,
                        right=assignee_hhi_cpc, how="outer", on=["assignee_id", "year"])

# create full panel
firm_list = assignee_hhi["assignee_id"].unique()
n_years = 44
ids_full = np.array([[x]*n_years for x in firm_list]).flatten()
dates = list(range(1976, 2020)) * len(firm_list)
balanced_panel = pd.DataFrame({"assignee_id": ids_full, "year": dates})
assignee_hhi_balanced = pd.merge(left=balanced_panel, right=assignee_hhi,  on=["assignee_id", "year"], how="left")

# merge with organization
assignee = pd.read_csv(
    "PatentsView/assignee.tsv", sep="\t", usecols=["id", "type", "organization"])
assignee = assignee.drop_duplicates()
assignee = assignee.rename(columns={"id": "assignee_id"})
assignee.loc[assignee["type"] == 2, "assignee_type"] = "US Company or Corporation"
assignee.loc[assignee["type"] == 3, "assignee_type"] = "Foreign Company or Corporation"
assignee = assignee.drop(columns="type")
assignee_hhi_balanced = pd.merge(left=assignee_hhi_balanced, right=assignee, how="left", on="assignee_id")


########
# save #
########
print("Finished, saving")
assignee_hhi_balanced.to_csv(
    "hdp_data_app/4_cpc_benchmarking/assignee_hhi_diversification_measures.csv", index=False)
