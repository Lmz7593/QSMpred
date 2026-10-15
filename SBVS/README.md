# SBVS
Here we provide a model for quantitatively predicting QS protein-ligand affinity.

# Data Structure
For base_train, the data structure follows the format provided in the qs/input.
For the test set, the data structure is shown below:

__Data Structure__

| CID | smiles | receptor | protein | 
|:----------:|:----------:|:----------:|:----------:| 
| 100016 | O=c1ncccn1[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O | CviR | MVIS....AGL | 
| 10007 | CC(C)(N)Cc1ccc(Cl)cc1 | LasR | MPYKA....FGV | 
| ... | ... | ... | ...| 
| 100094 | N[C@@H](CCC(=O)O)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O | LsrB | MALV....DSL | 

# Using of QSMpred
For each model, the following steps should be followed to use the model.
First, you should activate the environment:
```
conda activate QSMpred
```
## preprocess data
```
python 01_dataprocess.py --mode train --input_csv data/qs/input.csv --use_bert --esm_layers 29,30
```
```
python 01_dataprocess.py --mode test --input_csv data/test/test.csv --use_bert --esm_layers 29,30
```

## train model
```
python 03_train_model.py --exp_name qspred
```

## test
```
python 04_predict.py --exp_name qspred
```