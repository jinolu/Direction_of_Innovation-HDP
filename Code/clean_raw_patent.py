import pandas as pd
from tqdm import tqdm

patent = pd.read_csv("raw_data/patent.tsv", sep="\t")
patent["year"] = patent["date"].str[:4]

patent1 = patent[patent["date"].isnull() == False]
patent1 = patent1[patent1["type"] == "utility"]
patent1["date"] = pd.to_datetime(patent1["date"])
patent1["year"] = patent1["date"].dt.year
patent1 = patent1.drop_duplicates()
patent1 = patent1.sort_values(by="date", ascending=False)

# strip, drop nan and keep patents that have more than 5 words.
patent1["abstract"] = patent1["abstract"].str.strip()
patent1 = patent1[patent1["abstract"].isnull() == False]
patent1 = patent1[patent1["abstract"].str.count(" ") > 5]


for year in tqdm(range(2000, 2019)):
    patent2 = patent1[patent1["year"] == year]
    patent2.to_csv("./hdp_data/cleaned_text/"+str(year)+".csv", index=False, header=False)
    id = patent2["id"]
    id.to_csv("./hdp_data/paper_ids/"+str(year)+"_docID.txt", index=False)