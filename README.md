# text_gcn.pytorch

This project implements ["Graph Convolutional Networks for Text Classification. Yao et al. AAAI2019."](https://arxiv.org/abs/1809.05679) in PyTorch.

This implementation highly based on official code [yao8839836/text_gcn](<https://github.com/yao8839836/text_gcn>).

## Require

* Python 3.6
* PyTorch 1.0

## Running training and evaluation

1. `cd ./preprocess`
2. Run `python remove_words.py <dataset>`
3. Run `python build_graph.py <dataset>`
4. `cd ..`
5. Run `python train.py <dataset>`
6. Replace `<dataset>` with `20ng`, `R8`, `R52`, `ohsumed` or `mr`

## Visualization

### R8

<div align="center">    
<img src="https://github.com/iworldtong/text_gcn.pytorch/blob/master/src/R8_gcn_test.png?raw=true" width="500px" height="400px" alt="R8_gcn_test" align=center />
</div>

