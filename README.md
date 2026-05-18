# QSMpred
A LBVS classifier for mining Quorum Sensing Molecules

# QSMpred Environment Setup for Local Deployment
QSMpred is validated for Windows. We recommend installation via conda. For set up using conda, please refer to the relevant code line provided below.
conda env create -f environment.yml

# Data Structure
For base_train, the data structure follows the format provided in the file.
For the test set, the data structure is shown below:

__Data Structure__

| ID | smiles |
|:----------:|:----------:|:----------:| 
| 100016 | O=c1ncccn1[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O |
| 10007 | CC(C)(N)Cc1ccc(Cl)cc1 |
| ... | ... |
| 100094 | N[C@@H](CCC(=O)O)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O |

# Using of QSMpred
For each model, the following steps should be followed to use the model.
## preprocess data
python 01_datapreprocess.py

## train model
python 02_train_models.py

## test
python 03_test.py
