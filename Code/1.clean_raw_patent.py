import pandas as pd
from tqdm import tqdm

patent = pd.read_csv("PatentsView/patent.tsv", sep="\t")
app = pd.read_csv("PatentsView/application.tsv", sep="\t", usecols=["patent_id", "date"])
app["app_year"] = app["date"].str[:4]
app["app_year"] = app["app_year"].astype("int")
app = app[["patent_id", "app_year"]]
app = app.rename(columns={"patent_id": "id"})

# clean patent. strip, drop nan and keep patents that have more than 5 words.
patent = patent[patent["type"] == "utility"]
patent["abstract"] = patent["abstract"].str.strip()
patent = patent[patent["abstract"].isnull() == False]
patent = patent[patent["abstract"].str.count(" ") > 5]
patent = patent.drop_duplicates()

# app
patent["id"] = patent["id"].astype("str")
app["id"] = app["id"].astype("str")

# merge
patent = pd.merge(left=patent, right=app, how="left", on="id")
patent = patent.drop_duplicates()
patent = patent.sort_values(by="app_year", ascending=False)

patent['text'] = patent['title'] + ". " + patent['abstract']

for year in tqdm(range(1976, 2020)):
    patent1 = patent[patent["app_year"] == year]
    patent1.to_csv("./hdp_data_app/0_cleaned_text/"+str(year)+".csv", index=False)
