#!/bin/bash

for ((year=2008; year<=2011;year+=1)); do
  ./hdp --verbose --train_data /Users/LuJinhong\ 1\ 2/Desktop/patent_method/hdp_data/preprocessed_vector/$year.dat --directory /Users/LuJinhong\ 1\ 2/Desktop/patent_method/hdp_data/HDP_results/$year
done
