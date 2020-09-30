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
    data['categories'] = data['categories'].str.split(',')
    data['text'] = data.text.apply(lambda x : x.replace('\n',''))

    text = data['text'].values
    np.savetxt('../data/corpus/yelp.txt', text, delimiter=" ", fmt="%s", newline="\n")
    
    label = pd.read_csv("../data/corpus/yelp_label.csv", names=['category', 'label'], encoding='latin1')
    label_list = label.category.values.tolist()
    category = ['Active Life','Arts & Entertainment','Automotive','Beauty & Spas','Education','Event Planning & Services','Financial Services','Food','Health & Medical','Home Services','Hotels & Travel','Local Flavor','Local Services','Mass Media','Nightlife','Pets','Professional Services','Public Services & Government','Real Estate','Religious Organizations','Restaurants','Shopping']
    
    data['label'] = data['categories'].apply(lambda x : list(set(x) & set(label_list))).values
    data['label'] = data['label'].apply(lambda x: label.loc[label.category==x[0]].label.values)
    data['label'] = data['label'].apply(lambda x: int(x[0]))
    data['label'] = data['label'].apply(lambda x: category[x-1])

    trainIdx = data.sample(frac=0.7, replace=False).index
    train = data.loc[trainIdx].reset_index()
    train['set'] = 'train'
    train = train[['set','label']]
    test = data.loc[~data.index.isin(trainIdx)].reset_index()
    test['set'] = 'test'
    test = test[['set','label']]
    label_txt = pd.concat([train, test], axis=0).reset_index().reset_index()
    label_txt = label_txt[['level_0','set','label']]

    np.savetxt('../data/yelp.txt', label_txt, delimiter="\t", fmt="%s")  