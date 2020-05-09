import sys
import numpy as np
import pandas as pd
import json 
sys.path.append('../')

if len(sys.argv) != 2:
	sys.exit("Use: python doc_transform.py <dataset>")

datasets = ['yelp']
dataset = sys.argv[1]

if dataset not in datasets:
	sys.exit("wrong dataset name")

# Check dataset file 
# Transform the file into txt file to preprocess 
# 1. raw txt file 
# 2. label txt file 
    
if dataset == 'yelp' :
    business = pd.DataFrame([json.loads(line) for line in open('../data/corpus/business.json', 'r', errors='ignore')])
    review = pd.DataFrame([json.loads(line) for line in open('../data/corpus/review.json', 'r', errors='ignore')])
    data = pd.merge(review, business)[['text', 'categories']]
    text = data['text'].values
    np.savetxt('yelp.txt', text, delimiter=" ", fmt="%s") 
    

