from ggplot import *
import pandas as pd
from tqdm import tqdm
import numpy as np
import re
pd.options.display.max_columns = 100
pd.options.display.max_rows = 1000

doc_topics = pd.read_csv("doc_topics.csv", usecols=["assignee_id", "year", "patent_id"])
patent_class = pd.read_csv("./raw_data/cpc_current.tsv", sep="\t",
                           usecols=["patent_id",  "subsection_id", "group_id"])
assignee_type = pd.read_csv("assignee_type.csv")

# merge doc_topics with patent_class first before merging with assignee_type
doc_topics = doc_topics.drop_duplicates()
doc_topics["patent_id"] = doc_topics["patent_id"].astype("str")
patent_class["patent_id"] = patent_class["patent_id"].astype("str")

assignee_class = pd.merge(left=doc_topics, right=patent_class, how="left", left_on="patent_id", right_on="patent_id")
assignee_class = assignee_class[assignee_class["subsection_id"].isnull() == False]

assignee_class = assignee_class.drop_duplicates().sort_values(by=[
    "assignee_id", "year"]).reset_index(drop=True)
#assignee_class.to_csv("assignee_class.csv", index=False)


# Herfindahl index and Euclidian index
def hhi(series):
    _, cnt = np.unique(series, return_counts=True)
    return np.square(cnt/cnt.sum()).sum()


assignee_year = assignee_class.groupby(["assignee_id", "year"])["patent_id", "subsection_id"].agg(
    {'patent_id': 'count', 'subsection_id': hhi}).reset_index()
assignee_year = assignee_year.rename(columns={"patent_id": "n_patent", "subsection_id": "hhi"})

# some companies only have 1 patent in some year, so their HHI is 1
assignee_year["euclidian"] = assignee_year["hhi"] ** 0.5
assignee_year["diversification"] = 1-assignee_year["euclidian"]


########################
# plot diversification #
########################
# merge with assignee_type
assignee_year = pd.merge(left=assignee_year, right=assignee_type,
                         left_on="assignee_id", right_on="assignee_id", how="left")
assignee_year.to_csv("assignee_year_alternative.csv", index=False)

# "software", "cryptography", "robots", "nanotechnology", "semiconductors", "pharmaceuticals", "AI"
# Then caculate the average by year
assignee_year = pd.read_csv("assignee_year_alternative.csv")
software_diver = assignee_year[assignee_year["software"] == 1].groupby("year")["diversification"].mean().reset_index()
software_diver["type"] = "software"

cryptography_diver = assignee_year[assignee_year["cryptography"]
                                   == 1].groupby("year")["diversification"].mean().reset_index()
cryptography_diver["type"] = "cryptography"

robots_diver = assignee_year[assignee_year["robots"] == 1].groupby("year")["diversification"].mean().reset_index()
robots_diver["type"] = "robots"

nanotechnology_diver = assignee_year[assignee_year["nanotechnology"]
                                     == 1].groupby("year")["diversification"].mean().reset_index()
nanotechnology_diver["type"] = "nanotechnology"

semiconductors_diver = assignee_year[assignee_year["semiconductors"]
                                     == 1].groupby("year")["diversification"].mean().reset_index()
semiconductors_diver["type"] = "semiconductors"

pharmaceuticals_diver = assignee_year[assignee_year["pharmaceuticals"]
                                      == 1].groupby("year")["diversification"].mean().reset_index()
pharmaceuticals_diver["type"] = "pharmaceuticals"

AI_diver = assignee_year[assignee_year["AI"]
                         == 1].groupby("year")["diversification"].mean().reset_index()
AI_diver["type"] = "AI"


diver_year = software_diver.append(cryptography_diver).append(robots_diver).append(
    nanotechnology_diver).append(semiconductors_diver).append(pharmaceuticals_diver).append(AI_diver)
#diver_year = diver_year[~diver_year["year"].isin([2000, 2001, 2002, 2003, 2004, 2005])]
# diversification plot
"""
diversification_plot = ggplot(aes(x="year", y="diversification", color="type"), data=diver_year) +\
    geom_line() + theme(axis_text_x=element_text(angle=45, hjust=1)) + \
    scale_x_continuous(breaks=range(2000, 2019)) + ggtitle("Diversification of Patent Class")
diversification_plot.save("./plot/patent_class_diversification.png")
"""
type = diver_year["type"].unique().tolist()
for i in type:
    dataset = diver_year[diver_year["type"] == i]
    if i != "AI":
        i = i.capitalize()
    diversification_plot = ggplot(aes(x="year", y="diversification"), data=dataset) + geom_line() + theme(
        axis_text_x=element_text(angle=45, hjust=1)) + scale_x_continuous(breaks=range(2000, 2019)) + ggtitle("Patent class-based measure of diversification: "+i)
    diversification_plot.save("./plot/Plots_all_years/patent_class_diversification_" + i + ".png")


#########################
# Number of new classes #
#########################
assignee_class = pd.read_csv("assignee_class.csv")
assignee_year_class = assignee_class.groupby([
    "assignee_id", "year"])["subsection_id"].unique().reset_index()
assignee_year_class["n_year"] = assignee_year_class.groupby("assignee_id")["year"].transform("count")
assignee_year_class = assignee_year_class[assignee_year_class["n_year"] > 1]
assignee_year_class = assignee_year_class.sort_values(by=["assignee_id", "year"]).reset_index(drop=True)
assignee_year_class = assignee_year_class.rename(columns={"subsection_id": "year_class"})
assignee_list = assignee_year_class["assignee_id"].unique().tolist()


assignee_newclass = pd.DataFrame()
for assignee in tqdm(assignee_list):
    # assignee=assignee_list[0]
    subyear_class = assignee_year_class.loc[assignee_year_class["assignee_id"]
                                            == assignee, ["assignee_id", "year", "year_class"]]
    year_list = subyear_class["year"].tolist()
    min_year = min(year_list)
    max_year = max(year_list)
    for t in range(min_year, max_year+1):
        # t=2014
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

        prior3_class_lists = subyear_class.loc[subyear_class["year"].isin(prior_3years), "year_class"].tolist()
        prior3_classes = [i3 for sublist3 in prior3_class_lists for i3 in sublist3]

        prior5_class_lists = subyear_class.loc[subyear_class["year"].isin(prior_5years), "year_class"].tolist()
        prior5_classes = [i5 for sublist5 in prior5_class_lists for i5 in sublist5]

        class_t = subyear_class.loc[subyear_class["year"] == t, "year_class"].values[0].tolist()

        # calculate number of new classes compared to last 3 and 5 years
        n_newclass_3 = np.nan
        n_newclass_5 = np.nan
        if check3 == True:
            n_newclass_3 = 0
            for i in class_t:
                # i=class_t[0]
                if i not in prior3_classes:
                    n_newclass_3 = n_newclass_3+1
        if check5 == True:
            n_newclass_5 = 0
            for i in class_t:
                # i=class_t[0]
                if i not in prior5_classes:
                    n_newclass_5 = n_newclass_5+1

        subyear_class.loc[subyear_class["year"] == t, "newclass_t-3"] = n_newclass_3
        subyear_class.loc[subyear_class["year"] == t, "newclass_t-5"] = n_newclass_5
    assignee_newclass = assignee_newclass.append(subyear_class)


assignee_newclass = assignee_newclass.reset_index(drop=True)
assignee_newclass.to_csv("assignee_newclass.csv", index=False)


# run the following after the program done
#################
# plot newclass #
#################
# newclass
newclass = pd.read_csv("assignee_newclass.csv")
# merge with assignee_type
newclass = pd.merge(left=newclass, right=assignee_type,
                    left_on="assignee_id", right_on="assignee_id", how="left")


# calculate average by year
software_newclass = newclass[newclass["software"] == 1].groupby("year")[
    "newclass_t-3", "newclass_t-5"].agg({"newclass_t-3": "mean", "newclass_t-5": "mean"}).reset_index()
software_newclass["type"] = "software"

cryptography_newclass = newclass[newclass["cryptography"] == 1].groupby("year")[
    "newclass_t-3", "newclass_t-5"].agg({"newclass_t-3": "mean", "newclass_t-5": "mean"}).reset_index()
cryptography_newclass["type"] = "cryptography"

robots_newclass = newclass[newclass["robots"] == 1].groupby("year")[
    "newclass_t-3", "newclass_t-5"].agg({"newclass_t-3": "mean", "newclass_t-5": "mean"}).reset_index()
robots_newclass["type"] = "robots"

nanotechnology_newclass = newclass[newclass["nanotechnology"] == 1].groupby("year")[
    "newclass_t-3", "newclass_t-5"].agg({"newclass_t-3": "mean", "newclass_t-5": "mean"}).reset_index()
nanotechnology_newclass["type"] = "nanotechnology"

semiconductors_newclass = newclass[newclass["semiconductors"] == 1].groupby("year")[
    "newclass_t-3", "newclass_t-5"].agg({"newclass_t-3": "mean", "newclass_t-5": "mean"}).reset_index()
semiconductors_newclass["type"] = "semiconductors"

pharmaceuticals_newclass = newclass[newclass["pharmaceuticals"] == 1].groupby("year")[
    "newclass_t-3", "newclass_t-5"].agg({"newclass_t-3": "mean", "newclass_t-5": "mean"}).reset_index()
pharmaceuticals_newclass["type"] = "pharmaceuticals"

AI_newclass = newclass[newclass["AI"] == 1].groupby("year")[
    "newclass_t-3", "newclass_t-5"].agg({"newclass_t-3": "mean", "newclass_t-5": "mean"}).reset_index()
AI_newclass["type"] = "AI"


newclass_year = software_newclass.append(cryptography_newclass).append(robots_newclass).append(
    nanotechnology_newclass).append(semiconductors_newclass).append(pharmaceuticals_newclass).append(AI_newclass)
#newclass_year = newclass_year[~newclass_year["year"].isin([2000, 2001, 2002, 2003, 2004, 2005])]

# research newclass t-3 plot
"""
newclass_plot = ggplot(aes(x="year", y="newclass_t-3", color="type"), data=newclass_year) + geom_line() + theme(
    axis_text_x=element_text(angle=45, hjust=1)) + scale_x_continuous(breaks=range(2000, 2019)) + ggtitle("Research newclass: compared with t-3 years")
newclass_plot.save("./plot/newclass_t-3.png")
"""

for i in type:
    dataset = newclass_year[newclass_year["type"] == i]
    if i != "AI":
        i = i.capitalize()
    newclass_plot = ggplot(aes(x="year", y="newclass_t-3"), data=dataset) + geom_line() + theme(
        axis_text_x=element_text(angle=45, hjust=1)) + scale_x_continuous(breaks=range(2000, 2019)) + ggtitle("Patent class-based measure of trajectory: "+i)
    newclass_plot.save("./plot/Plots_all_years/patent_class_trajectory_" + i + ".png")


"""
# research newclass t-5 plot
newclass_plot = ggplot(aes(x="year", y="newclass_t-5", color="type"), data=newclass_year) + geom_line() + theme(
    axis_text_x=element_text(angle=45, hjust=1)) + scale_x_continuous(breaks=range(2000, 2019)) + ggtitle("Research newclass: compared with t-5 years")
newclass_plot.save("./plot/newclass_t-5.png")

newclass_plot = ggplot(aes(x="year", y="newclass_t-5", color="type"), data=newclass_year) + geom_line() + theme(
    axis_text_x=element_text(angle=45, hjust=1)) + scale_x_continuous(breaks=range(2000, 2019)) + ggtitle("Research newclass: compared with t-5 years")+facet_wrap("type", scales="free", nrow=3)
newclass_plot.save("./plot/newclass_t-5_group.png")
"""
