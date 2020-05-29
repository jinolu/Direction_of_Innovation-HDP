from ggplot import *
import pandas as pd
from tqdm import tqdm
import numpy as np
import re
pd.options.display.max_columns = 100
pd.options.display.max_rows = 1000

raw_assignee = pd.read_csv("./raw_data/patent_assignee.tsv", sep="\t")
patent_class = pd.read_csv("./raw_data/cpc_current.tsv", sep="\t", nrows=100)


#########################################
# trajectory measure of diversification #
#########################################
doc_topics = pd.DataFrame()
for i in tqdm(range(2000, 2019)):
    year = str(i)
    file = "./hdp_data/doc_topics_words/"+year+"/doc_topics_"+year+".csv"
    subdoc_topics = pd.read_csv(file, header=0, names=["patent_id", "topic_id", "probability"])
    subdoc_topics["year"] = i
    doc_topics = doc_topics.append(subdoc_topics)
doc_topics = doc_topics.reset_index(drop=True)
doc_topics["patent_id"].nunique()

doc_topics["patent_id"] = doc_topics["patent_id"].astype("str")
raw_assignee["patent_id"] = raw_assignee["patent_id"].astype("str")

# merge with raw_assignee
doc_topics = pd.merge(left=doc_topics, right=raw_assignee, how="left", left_on="patent_id", right_on="patent_id")

doc_topics = doc_topics[doc_topics["assignee_id"].isnull() == False]
# 4132983
# 327865 patents with missing patent_id need to drop those
doc_topics = doc_topics[doc_topics["assignee_id"].str.contains("per_") == False]
doc_topics = doc_topics[["assignee_id", "year", "patent_id", "topic_id", "probability"]]
doc_topics = doc_topics.sort_values(by=["assignee_id", "year", "patent_id", "topic_id"]).reset_index(drop=True)
doc_topics.to_csv("doc_topics.csv", index=False)

######################
# Calculate and plot #
######################
doc_topics = pd.read_csv("doc_topics.csv")
# calculate number of topics by assignee id by year
diver_assignee_year = doc_topics.groupby(["assignee_id", "year"])["topic_id", "patent_id"].agg(
    {"topic_id": {"n_topics": "count", "nuni_topics": "nunique"}, "patent_id": {"n_patent": "count"}}).reset_index()


# Then caculate the average by year
diver_year = diver_assignee_year.groupby("year")["n_topics", "nuni_topics"].agg(
    {"n_topics": "mean", "nuni_topics": "mean"}).reset_index()


#######################
# research trajectory #
#######################
# load topic_simiarity
doc_topics = pd.read_csv("doc_topics.csv")
topic_similarity = pd.read_csv("topic_similarity_panel.csv")
topic_similarity["ParentTopic"] = topic_similarity["ParentTopic"].str.extract('(\d+)', expand=False)
topic_similarity["ComparedToTopic"] = topic_similarity["ComparedToTopic"].str.extract('(\d+)', expand=False)
topic_similarity["ParentTopicYear"] = topic_similarity["ParentTopicYear"].astype("str")
topic_similarity["ComparedToTopicYear"] = topic_similarity["ComparedToTopicYear"].astype("str")
topic_similarity["key"] = topic_similarity["ParentTopicYear"]+"-" + \
    topic_similarity["ParentTopic"]+"-"+topic_similarity["ComparedToTopicYear"] + "-"+topic_similarity["ComparedToTopic"]
topic_similarity["Distance"] = 1-topic_similarity["IndexofSimilarity"]
tdistance_dict = topic_similarity.set_index("key")["Distance"].to_dict()

# calculate
doc_topics["year_topic"] = doc_topics["year"].astype("str")+"-"+doc_topics["topic_id"].astype("str")
assignee_year_topic = doc_topics.groupby(["assignee_id", "year"])["year_topic"].unique().reset_index()
assignee_year_topic["n_year"] = assignee_year_topic.groupby("assignee_id")["year"].transform("count")
assignee_year_topic = assignee_year_topic[assignee_year_topic["n_year"] > 1]
assignee_year_topic = assignee_year_topic.sort_values(by=["assignee_id", "year"]).reset_index(drop=True)
assignee_list = assignee_year_topic["assignee_id"].unique().tolist()

assignee_trajectory = pd.DataFrame()
for assignee in tqdm(assignee_list):
    # assignee=assignee_list[0]
    subyear_topic = assignee_year_topic.loc[assignee_year_topic["assignee_id"]
                                            == assignee, ["assignee_id", "year", "year_topic"]]
    year_list = subyear_topic["year"].tolist()
    min_year = min(year_list)
    max_year = max(year_list)
    for t in range(min_year, max_year+1):
        if t not in year_list:
            continue

        t_1 = t-1
        t_2 = t-2
        t_3 = t-3
        t_4 = t-4
        t_5 = t-5
        # t_1=2002
        # t_5=2014
        prior_3years = [t_1, t_2, t_3]
        prior_5years = [t_1, t_2, t_3, t_4, t_5]

        check3 = any(item in year_list for item in prior_3years)
        check5 = any(item in year_list for item in prior_5years)

        prior3_topic_lists = subyear_topic.loc[subyear_topic["year"].isin(prior_3years), "year_topic"].tolist()
        prior3_topics = [i3 for sublist3 in prior3_topic_lists for i3 in sublist3]

        prior5_topic_lists = subyear_topic.loc[subyear_topic["year"].isin(prior_5years), "year_topic"].tolist()
        prior5_topics = [i5 for sublist5 in prior5_topic_lists for i5 in sublist5]

        topic_t = subyear_topic.loc[subyear_topic["year"] == t, "year_topic"].values[0].tolist()

        # sum all the similarity pairs in t3,t5 and t through 2 nested loops
        n_topics_3 = len(prior3_topics)
        distance_sum_3 = 0

        n_topics_5 = len(prior5_topics)
        distance_sum_5 = 0

        distance_avg_3 = np.nan
        distance_avg_5 = np.nan

        if check3 == True:
            for i in prior3_topics:
                # i = prior3_topics[0]
                for j in topic_t:
                    # j=topic_t1[0]
                    topic_key = i+"-"+j
                    topic_distance = tdistance_dict[topic_key]
                    distance_sum_3 = distance_sum_3+topic_distance
            distance_avg_3 = distance_sum_3/n_topics_3
        if check5 == True:
            for i in prior5_topics:
                # i = prior5_topics[0]
                for j in topic_t:
                    # j=topic_t1[0]
                    topic_key = i+"-"+j
                    topic_distance = tdistance_dict[topic_key]
                    distance_sum_5 = distance_sum_5+topic_distance
            distance_avg_5 = distance_sum_5/n_topics_5

        subyear_topic.loc[subyear_topic["year"] == t, "distance_t-3"] = distance_avg_3
        subyear_topic.loc[subyear_topic["year"] == t, "distance_t-5"] = distance_avg_5
    assignee_trajectory = assignee_trajectory.append(subyear_topic)

assignee_trajectory = assignee_trajectory.reset_index(drop=True)
assignee_trajectory.to_csv("assignee_trajectory.csv", index=False)


assignee_trajectory = pd.read_csv("assignee_trajectory.csv")
# USPC Patent Class: 380 Cryptography, 901 Robotics and 977 Nanotechnology
uspc = pd.read_csv("raw_data/uspc_current.tsv", sep="\t")
uspc = uspc[["mainclass_id", "subclass_id"]].drop_duplicates().sort_values(
    by=["mainclass_id", "subclass_id"]).reset_index(drop=True)

uspc[["main", "subclass"]] = uspc['subclass_id'].str.split('/', expand=True)
uspc["uspc_subclass"] = uspc["subclass"]
uspc.loc[uspc["subclass"].str.contains("[A-Za-z]", na=False), "uspc_subclass"] = np.nan
uspc["uspc_subclass"] = uspc["uspc_subclass"].str.replace(" ", "")
uspc.loc[uspc["uspc_subclass"] == "", "uspc_subclass"] = np.nan
uspc["uspc_subclass"] = uspc["uspc_subclass"].astype("float")
uspc["uspc_subclass"] = uspc["uspc_subclass"]*1000
uspc["uspc_class"] = uspc["mainclass_id"]
uspc = uspc[["mainclass_id", "subclass_id", "uspc_class", "uspc_subclass"]]
uspc.to_csv("uspc.csv", index=False)

# load software patent
software = pd.read_csv("software.csv")
software = software[["mainclass_id", "subclass_id", "software_flag"]]
uspc = pd.read_csv("raw_data/uspc_current.tsv", sep="\t")
uspc = pd.merge(left=uspc, right=software, how="left", left_on=[
    "mainclass_id", "subclass_id"], right_on=["mainclass_id", "subclass_id"])
uspc["mainclass_id"] = uspc["mainclass_id"].astype("str")
uspc.loc[uspc["mainclass_id"] == "380", "cryptography"] = 1
uspc.loc[uspc["mainclass_id"] == "901", "robots"] = 1
uspc.loc[uspc["mainclass_id"] == "977", "nanotechnology"] = 1
uspc[["cryptography", "robots", "nanotechnology"]] = uspc[[
    "cryptography", "robots", "nanotechnology"]].fillna(0)
uspc_patent = uspc.groupby("patent_id")["software_flag", "cryptography", "robots", "nanotechnology"].max().reset_index()
uspc_patent["patent_id"] = uspc_patent["patent_id"].astype("str")

# CPC patent
cpc = pd.read_csv("./raw_data/cpc_current.tsv", sep="\t")
cpc.loc[cpc["group_id"] == "H01L", "semiconductors"] = 1
cpc.loc[cpc["group_id"] == "A61K", "pharmaceuticals"] = 1
cpc[["semiconductors", "pharmaceuticals"]] = cpc[["semiconductors", "pharmaceuticals"]].fillna(0)
cpc_patent = cpc.groupby("patent_id")["semiconductors", "pharmaceuticals"].max().reset_index()
cpc_patent["patent_id"] = cpc_patent["patent_id"].astype("str")


# patent_identifier
patent_type = pd.merge(left=uspc_patent, right=cpc_patent, left_on="patent_id", right_on="patent_id", how="outer")
patent_type = patent_type.rename(columns={"software_flag": "software"})
patent_type[["software", "cryptography", "robots", "nanotechnology", "semiconductors", "pharmaceuticals"]] = patent_type[[
    "software", "cryptography", "robots", "nanotechnology", "semiconductors", "pharmaceuticals"]].fillna(0).astype("int")

# identify AI patents
AI_patent = pd.read_csv("Classified Test Data 4.csv")
AI_patent["patent_id"] = AI_patent["id"].str.replace("US", "")
AI_patent["patent_id"] = AI_patent["patent_id"].str.replace("B2", "")
AI_patent["patent_id"] = AI_patent["patent_id"].str.replace("B1", "")
AI_patent["patent_id"] = AI_patent["patent_id"].str.replace("I5", "")
AI_patent["patent_id"] = AI_patent["patent_id"].str.replace("NA", "")
AI_patent["patent_id"] = AI_patent["patent_id"].str.replace("NA", "")
AI_patent.loc[AI_patent["id"].str.contains("US8182619Tire"), "patent_id"] = "8182619"
AI_patent.loc[AI_patent["id"].str.contains("US8217621Active"), "patent_id"] = "8217621"
AI_patent.loc[AI_patent["id"].str.contains("US8248399Image"), "patent_id"] = "8248399"
AI_patent.loc[AI_patent["id"].str.contains("US8288508Inosine"), "patent_id"] = "8288508"
AI_patent.loc[AI_patent["id"].str.contains("US8331281Method"), "patent_id"] = "8331281"
AI_patent.loc[AI_patent["id"].str.contains("US8341296Adjustable"), "patent_id"] = "8341296"
AI_patent = AI_patent.rename(columns={"is_ai_p": "AI"})
AI_patent["patent_id"] = AI_patent["patent_id"].astype("str")
AI_patent = AI_patent[["patent_id", "AI"]]
patent_type = pd.merge(patent_type, AI_patent, on="patent_id", how="left")
patent_type["AI"] = patent_type["AI"].fillna(0).astype("int")

# patent_assignee
raw_assignee = pd.read_csv("./raw_data/patent_assignee.tsv", sep="\t")
raw_assignee["patent_id"] = raw_assignee["patent_id"].astype("str")
raw_assignee = pd.merge(left=raw_assignee, right=patent_type, left_on="patent_id", right_on="patent_id", how="left")
raw_assignee[["software", "cryptography", "robots", "nanotechnology", "semiconductors", "pharmaceuticals", "AI"]] = raw_assignee[[
    "software", "cryptography", "robots", "nanotechnology", "semiconductors", "pharmaceuticals", "AI"]].fillna(0).astype("int")

assignee_type = raw_assignee.groupby("assignee_id")[
    "software", "cryptography", "robots", "nanotechnology", "semiconductors", "pharmaceuticals", "AI"].max().reset_index()
assignee_type.to_csv("assignee_type.csv", index=False)


#######################################
# load diversification and trajectory #
#######################################
assignee_type = pd.read_csv("assignee_type.csv")
# Diversification
doc_topics = pd.read_csv("doc_topics.csv")

# calculate number of topics by assignee id by year
diver_assignee_year = doc_topics.groupby(["assignee_id", "year"])["topic_id", "patent_id"].agg(
    {"topic_id": {"n_topics": "count", "nuni_topics": "nunique"}, "patent_id": {"n_patent": "count"}}).reset_index()

# merge with assignee_type
diver_assignee_year = pd.merge(left=diver_assignee_year, right=assignee_type,
                               left_on="assignee_id", right_on="assignee_id", how="left")


assignee_type["software"].sum()
assignee_type["cryptography"].sum()
assignee_type["robots"].sum()
assignee_type["nanotechnology"].sum()
assignee_type["semiconductors"].sum()
assignee_type["pharmaceuticals"].sum()
assignee_type["AI"].sum()
assignee_type["assignee_id"].nunique()

# "software", "cryptography", "robots", "nanotechnology", "semiconductors", "pharmaceuticals"
# Then caculate the average by year
software_diver = diver_assignee_year[diver_assignee_year["software"] == 1].groupby("year")[
    "n_topics", "nuni_topics"].agg({"n_topics": "mean", "nuni_topics": "mean"}).reset_index()
software_diver["type"] = "software"

cryptography_diver = diver_assignee_year[diver_assignee_year["cryptography"] == 1].groupby("year")[
    "n_topics", "nuni_topics"].agg({"n_topics": "mean", "nuni_topics": "mean"}).reset_index()
cryptography_diver["type"] = "cryptography"

robots_diver = diver_assignee_year[diver_assignee_year["robots"] == 1].groupby("year")[
    "n_topics", "nuni_topics"].agg({"n_topics": "mean", "nuni_topics": "mean"}).reset_index()
robots_diver["type"] = "robots"

nanotechnology_diver = diver_assignee_year[diver_assignee_year["nanotechnology"] == 1].groupby("year")[
    "n_topics", "nuni_topics"].agg({"n_topics": "mean", "nuni_topics": "mean"}).reset_index()
nanotechnology_diver["type"] = "nanotechnology"

semiconductors_diver = diver_assignee_year[diver_assignee_year["semiconductors"] == 1].groupby("year")[
    "n_topics", "nuni_topics"].agg({"n_topics": "mean", "nuni_topics": "mean"}).reset_index()
semiconductors_diver["type"] = "semiconductors"

pharmaceuticals_diver = diver_assignee_year[diver_assignee_year["pharmaceuticals"] == 1].groupby("year")[
    "n_topics", "nuni_topics"].agg({"n_topics": "mean", "nuni_topics": "mean"}).reset_index()
pharmaceuticals_diver["type"] = "pharmaceuticals"

AI_diver = diver_assignee_year[diver_assignee_year["AI"] == 1].groupby("year")[
    "n_topics", "nuni_topics"].agg({"n_topics": "mean", "nuni_topics": "mean"}).reset_index()
AI_diver["type"] = "AI"


diver_year = software_diver.append(cryptography_diver).append(robots_diver).append(
    nanotechnology_diver).append(semiconductors_diver).append(pharmaceuticals_diver).append(AI_diver)
#diver_year = diver_year[~diver_year["year"].isin([2000, 2001, 2002, 2003, 2004, 2005])]
# intensive plot
"""
intensive_plot = ggplot(aes(x="year", y="n_topics", color="type"), data=diver_year) + geom_line() + theme(
    axis_text_x=element_text(angle=45, hjust=1)) + scale_x_continuous(breaks=range(2000, 2019)) + ggtitle("Intensive Measure of Diversification")
intensive_plot.save("./plot/intensive.png")
"""

type = diver_year["type"].unique().tolist()
for i in type:
    dataset = diver_year[diver_year["type"] == i]
    if i != "AI":
        i = i.capitalize()
    intensive_plot = ggplot(aes(x="year", y="n_topics"), data=dataset) + geom_line() + theme(
        axis_text_x=element_text(angle=45, hjust=1)) + scale_x_continuous(breaks=range(2000, 2019)) + ggtitle("HDP-based intensive measure of diversification: "+i)
    intensive_plot.save("./plot/Plots_all_years/HDP_intensive_" + i + ".png")


# extensive plot
"""
extensive_plot = ggplot(aes(x="year", y="nuni_topics", color="type"), data=diver_year) + geom_line() + theme(axis_text_x=element_text(
    angle=45, hjust=1)) + scale_x_continuous(breaks=range(2000, 2019)) + ggtitle("Extensive Measure of Diversification")
extensive_plot.save("./plot/extensive.png")
"""

for i in type:
    dataset = diver_year[diver_year["type"] == i]
    if i != "AI":
        i = i.capitalize()
    extensive_plot = ggplot(aes(x="year", y="nuni_topics"), data=dataset) + geom_line() + theme(
        axis_text_x=element_text(angle=45, hjust=1)) + scale_x_continuous(breaks=range(2000, 2019)) + ggtitle("HDP-based extensive measure of diversification: "+i)
    extensive_plot.save("./plot/Plots_all_years/HDP_extensive_" + i + ".png")

# Trajectory
trajectory = pd.read_csv("assignee_trajectory.csv")
# merge with assignee_type
trajectory = pd.merge(left=trajectory, right=assignee_type,
                      left_on="assignee_id", right_on="assignee_id", how="left")


# calculate average by year
software_traject = trajectory[trajectory["software"] == 1].groupby("year")[
    "distance_t-3", "distance_t-5"].agg({"distance_t-3": "mean", "distance_t-5": "mean"}).reset_index()
software_traject["type"] = "software"

cryptography_traject = trajectory[trajectory["cryptography"] == 1].groupby("year")[
    "distance_t-3", "distance_t-5"].agg({"distance_t-3": "mean", "distance_t-5": "mean"}).reset_index()
cryptography_traject["type"] = "cryptography"

robots_traject = trajectory[trajectory["robots"] == 1].groupby("year")[
    "distance_t-3", "distance_t-5"].agg({"distance_t-3": "mean", "distance_t-5": "mean"}).reset_index()
robots_traject["type"] = "robots"

nanotechnology_traject = trajectory[trajectory["nanotechnology"] == 1].groupby("year")[
    "distance_t-3", "distance_t-5"].agg({"distance_t-3": "mean", "distance_t-5": "mean"}).reset_index()
nanotechnology_traject["type"] = "nanotechnology"

semiconductors_traject = trajectory[trajectory["semiconductors"] == 1].groupby("year")[
    "distance_t-3", "distance_t-5"].agg({"distance_t-3": "mean", "distance_t-5": "mean"}).reset_index()
semiconductors_traject["type"] = "semiconductors"

pharmaceuticals_traject = trajectory[trajectory["pharmaceuticals"] == 1].groupby("year")[
    "distance_t-3", "distance_t-5"].agg({"distance_t-3": "mean", "distance_t-5": "mean"}).reset_index()
pharmaceuticals_traject["type"] = "pharmaceuticals"

AI_traject = trajectory[trajectory["AI"] == 1].groupby("year")[
    "distance_t-3", "distance_t-5"].agg({"distance_t-3": "mean", "distance_t-5": "mean"}).reset_index()
AI_traject["type"] = "AI"

traject_year = software_traject.append(cryptography_traject).append(robots_traject).append(
    nanotechnology_traject).append(semiconductors_traject).append(pharmaceuticals_traject).append(AI_traject)
#traject_year = traject_year[~traject_year["year"].isin([2000, 2001, 2002, 2003, 2004, 2005])]
# research trajectory t-3 plot
"""
trajectory_plot = ggplot(aes(x="year", y="distance_t-3", color="type"), data=traject_year) + geom_line() + theme(
    axis_text_x=element_text(angle=45, hjust=1)) + scale_x_continuous(breaks=range(2000, 2019)) + ggtitle("Research Trajectory: compared with t-3 years")
trajectory_plot.save("./plot/trajectory_t-3.png")
"""

for i in type:
    dataset = traject_year[traject_year["type"] == i]
    if i != "AI":
        i = i.capitalize()
    trajectory_plot = ggplot(aes(x="year", y="distance_t-3"), data=dataset) + geom_line() + theme(
        axis_text_x=element_text(angle=45, hjust=1)) + scale_x_continuous(breaks=range(2000, 2019)) + ggtitle("HDP-based measure of trajectory: "+i)
    trajectory_plot.save("./plot/Plots_all_years/HDP_trajectory_" + i + ".png")


"""
# research trajectory t-5 plot
trajectory_plot = ggplot(aes(x="year", y="distance_t-5", color="type"), data=traject_year) + geom_line() + theme(
    axis_text_x=element_text(angle=45, hjust=1)) + scale_x_continuous(breaks=range(2000, 2019)) + ggtitle("Research Trajectory: compared with t-5 years")
trajectory_plot.save("./plot/trajectory_t-5.png")

trajectory_plot = ggplot(aes(x="year", y="distance_t-5", color="type"), data=traject_year) + geom_line() + theme(
    axis_text_x=element_text(angle=45, hjust=1)) + scale_x_continuous(breaks=range(2000, 2019)) + ggtitle("Research Trajectory: compared with t-5 years")+facet_wrap("type", scales="free", nrow=3)
trajectory_plot.save("./plot/trajectory_t-5_group.png")

# test correlation
diver_assignee_year["any"] = diver_assignee_year[["software", "cryptography", "robots", "nanotechnology",
                                                  "semiconductors", "pharmaceuticals"]].max(axis=1)

correlation = diver_assignee_year[diver_assignee_year["any"] == 1]
correlation = correlation[["software", "cryptography", "robots",
                           "nanotechnology", "semiconductors", "pharmaceuticals"]].corr(method="pearson")

# trajectory
trajectory["any"] = trajectory[["software", "cryptography", "robots", "nanotechnology",
                                "semiconductors", "pharmaceuticals"]].max(axis=1)

correlation = diver_assignee_year[diver_assignee_year["any"] == 1]
correlation = correlation[["software", "cryptography", "robots",
                           "nanotechnology", "semiconductors", "pharmaceuticals"]].corr(method="pearson")


correlation.to_csv("correlation.csv")
"""
