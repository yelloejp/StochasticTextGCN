'''
emsemble smaller size of graphs : averaging? voting? 
Accuracy is around 85%....Voting is 55%
'''
from __future__ import division
from __future__ import print_function
from sklearn import metrics
import scipy.sparse as sp
from scipy.stats import mode
import random
import time
import sys
import os
import torch
import torch.nn as nn
import numpy as np
from utils.utils import *
from models.gcn import GCN
from models.mlp import MLP
from config import CONFIG
cfg = CONFIG()

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr', 'yelp']
dataset = sys.argv[1]
num_graph = int(sys.argv[2])

if dataset not in datasets:
	sys.exit("wrong dataset name")
cfg.dataset = dataset

# Set random seed
seed = random.randint(1, 200)
seed = 2019
np.random.seed(seed)
torch.manual_seed(seed)

# Load data
''' Description
- adj : adjacent matrix for all nodes
- features : vector embedding (it will be set to itentity matrix)'''
for i in range(num_graph):

    globals()['adj{}'.format(i)], \
    globals()['features{}'.format(i)], \
    globals()['y_train{}'.format(i)], \
    globals()['y_val{}'.format(i)], \
    globals()['y_test{}'.format(i)], \
    globals()['train_mask{}'.format(i)], \
    globals()['val_mask{}'.format(i)], \
    globals()['test_mask{}'.format(i)], \
    globals()['train_size{}'.format(i)], \
    globals()['test_size{}'.format(i)] = load_corpus(cfg.dataset, i)
    
    globals()['features{}'.format(i)] = sp.identity(globals()['features{}'.format(i)].shape[0])
    
models = []
for i in range(num_graph): 
    # Some preprocessing
    # Here preprocess adj(adjacent matrix) to adj_hat(called support)
    adj = globals()['adj{}'.format(i)]
    features = globals()['features{}'.format(i)]
    y_train = globals()['y_train{}'.format(i)]
    y_val = globals()['y_val{}'.format(i)]
    y_test = globals()['y_test{}'.format(i)]
    train_mask = globals()['train_mask{}'.format(i)]
    val_mask = globals()['val_mask{}'.format(i)]
    test_mask = globals()['test_mask{}'.format(i)]
    train_size = globals()['train_size{}'.format(i)]
    test_size = globals()['test_size{}'.format(i)]

    features = preprocess_features(features)

    if cfg.model == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN
    elif cfg.model == 'gcn_cheby':
        support = chebyshev_polynomials(adj, cfg.max_degree)
        num_supports = 1 + cfg.max_degree
        model_func = GCN
    elif cfg.model == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for model: ' + str(cfg.model))

    # Define placeholders
    t_features = torch.from_numpy(features)
    t_y_train = torch.from_numpy(y_train)
    t_y_val = torch.from_numpy(y_val)
    t_y_test = torch.from_numpy(y_test)
    t_train_mask = torch.from_numpy(train_mask.astype(np.float32))
    tm_train_mask = torch.transpose(torch.unsqueeze(t_train_mask, 0), 1, 0).repeat(1, y_train.shape[1])
          
    t_support = []

    for i in range(len(support)):
        t_support.append(torch.Tensor(support[i]))
    
    model = model_func(input_dim=features.shape[0], support=t_support, num_classes=y_train.shape[1])

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # Define model evaluation function
    def evaluate(features, labels, mask):
        t_test = time.time()
        model.eval()
        with torch.no_grad():
            logits = model(features)
            t_mask = torch.from_numpy(np.array(mask*1., dtype=np.float32))
            tm_mask = torch.transpose(torch.unsqueeze(t_mask, 0), 1, 0).repeat(1, labels.shape[1])
            loss = criterion(logits * tm_mask, torch.max(labels, 1)[1])
            pred = torch.max(logits, 1)[1]
            acc = ((pred == torch.max(labels, 1)[1]).float() * t_mask).sum().item() / t_mask.sum().item()
            
        return loss.numpy(), acc, pred.numpy(), labels.numpy(), (time.time() - t_test)
    
    val_losses = []

    # Train model
    start_train = time.time()

    for epoch in range(cfg.epochs): #cfg.epochs
        t = time.time()        
        # Forward pass
        logits = model(t_features)
        loss = criterion(logits * tm_train_mask, torch.max(t_y_train, 1)[1])   
        acc = ((torch.max(logits, 1)[1] == torch.max(t_y_train, 1)[1]).float() * t_train_mask).sum().item() / t_train_mask.sum().item()
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # Validation
        val_loss, val_acc, pred, labels, duration = evaluate(t_features, t_y_val, val_mask)
        val_losses.append(val_loss)
        
        print_log("Epoch: {:.0f}, train_loss= {:.5f}, train_acc= {:.5f}, val_loss= {:.5f}, val_acc= {:.5f}, time= {:.5f}"\
                    .format(epoch + 1, loss, acc, val_loss, val_acc, time.time() - t))
        
        if epoch > cfg.early_stopping and val_losses[-1] > np.mean(val_losses[-(cfg.early_stopping+1):-1]):
            print_log("Early stopping...")
            break
        
    end_train = time.time()  
    models.append(model)     
    print_log("Train runtime={:.5f}".format(end_train-start_train))

def test_evaluate(m, features, labels, mask):
    t_test = time.time()
    m.eval()
    
    with torch.no_grad():
        logits = m(features)
        if logits.shape[0] > mask.shape[0]:
            logits = logits[0:mask.shape[0]]
        elif logits.shape[0] < mask.shape[0]:
            nums = mask.shape[0] - logits.shape[0]
            temp_logits = logits[0:nums]
            logits = torch.cat((logits,temp_logits))

        t_mask = torch.from_numpy(np.array(mask*1., dtype=np.float32))        
        tm_mask = torch.transpose(torch.unsqueeze(t_mask, 0), 1, 0).repeat(1, labels.shape[1])        
        loss = criterion(logits * tm_mask, torch.max(labels, 1)[1])        
        pred = torch.max(logits, 1)[1]        
        acc = ((pred == torch.max(labels, 1)[1]).float() * t_mask).sum().item() / t_mask.sum().item()
        
    return loss.numpy(), acc, pred.numpy(), labels.numpy(), (time.time() - t_test)
    
#test_loss, test_acc, pred, labels, test_duration = evaluate(t_features, t_y_test, test_mask)
#print_log("Test set results: \n\t loss= {:.5f}, accuracy= {:.5f}, time= {:.5f}".format(test_loss, test_acc, test_duration))


# Testing
pred_hist = []
for m in models : 
    test_loss, test_acc, pred, labels, test_duration = test_evaluate(m, t_features, t_y_test, test_mask)
    pred_hist.append(pred)
    #print_log("Test set results: \n\t loss= {:.5f}, accuracy= {:.5f}, time= {:.5f}".format(test_loss, test_acc, test_duration))

agg_pred = np.zeros(len(pred))
#print(len(pred))
for m in range(num_graph):
    agg_pred += pred_hist[m]

doc_node = t_y_test.numpy()[test_mask]
doc_node = np.argmax(doc_node, axis=1)

avg_doc = agg_pred[test_mask] 
avg_comp = (avg_doc == doc_node) 
avg_acc = ((avg_comp==False).sum() /len(doc_node))

vote_node = mode(pred_hist, 0)[0][0]
vote_doc = vote_node[test_mask]
vote_comp = (vote_doc == doc_node) 
vote_acc = ((vote_comp==False).sum() /len(doc_node))

print_log("Overall averaging accuracy={:.5f}".format(avg_acc))
print_log("Overall voting accuracy={:.5f}".format(vote_acc))



