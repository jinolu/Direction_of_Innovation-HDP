from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from ggplot import *
from tqdm import tqdm
from plotnine import *
import numpy as np
import math

pd.options.display.max_rows = 500
run = pd.read_csv("hdp_data_app/2_HDP_results/hdp_run.csv")
run_likelihood = run[["iter", "likelihood"]]
run_likelihood = run_likelihood.rename(columns={"likelihood": "score"})
run_likelihood["type"] = "Likelihood"
run_topic = run[["iter", "num.topics"]]
run_topic = run_topic.rename(columns={"num.topics": "score"})
run_topic["type"] = "Number of Topics"
run_score = run_likelihood.append(run_topic)

run_plot = ggplot(aes(x="iter", y="score"), data=run_score) + \
    geom_line() + theme(figure_size=(25, 14), axis_text_x=element_text(angle=45)) + \
    ggtitle("HDP run with sample hyper turned off") + \
    scale_x_continuous(breaks=range(0, 1300, 100)) +\
    facet_wrap("type", scales="free", nrow=3) + ylab("")
run_plot.save("hdp_data_app/2_HDP_results/hdp_iterations.png")