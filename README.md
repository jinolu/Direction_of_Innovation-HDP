# Direction_of_Innovation-HDP
Code and data for "Teodoridis, Florenta and Lu, Jino and Furman, Jeffrey L., Measuring the Direction of Innovation: Frontier Tools in Unassisted Machine Learning" https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3596233

Introduce an approach based on an unassisted machine learning technique, Hierarchical Dirichlet Process (HDP), that flexibly generates categories from large scale text data and enables calculations of the distance and movement in ideas space.

We use USPTO patent data from PatentsView (https://www.patentsview.org/download/)

Please cite David Blei's work (http://www.cs.columbia.edu/~blei/). The code builds on topic modeling developed by his lab.

This code is built on the code in https://github.com/florentta/Economics_of_Innovation

Output files:

(1) assignee_hhi_diversification_measures.csv: This file contains the HDP- and CPC-based Herfindahl-Hirschman index (HHI) diversification measures for each USPTO assignee in each year between 1976 and 2019
- hhi_hdp: the HDP-based HHI concentration measure
- hhi_diversity_hdp: the HDP-based HHI diversification measure (i.e., 1-hhi_hdp)
- hhi_cpc:the CPC-based HHI concentration measure
- hhi_diversity_cpc: this is the CPC-based HHI diversification measure (i.e., 1-hhi_cpc)
