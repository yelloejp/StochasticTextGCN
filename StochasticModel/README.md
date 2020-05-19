# Note : Source codes will be uploaded gradually.

### Step 1) Transform dataset into relevant format. 
#### `cd ./preprocess`
#### Run `python doc_transform.py <dataset>`
    1) row text data -> ./data/corpus/dataset.txt 
    2) label data -> ./data/dataset.txt

### Step 2) Remove stop words from datasets.
#### Run `python remove_word.py <dataset>`

### Step 3) Build words and docs graph.
#### Run `python build_graph.py <dataset>`

### Step 4) Train the model. 
#### `cd ..`
#### Run `python train.py <dataset>`
