# QSMpred
we developed an efficient screening framework, QSMpred, which integrates LBVS and SBVS within a unified pipeline to discover potential QSMs. For LBVS, we constructed a series of classifiers based on molecular structure, combining traditional machine learning algorithms with deep learning architectures. For SBVS, we built a model that predicts protein–ligand affinity quantitatively from sequence and SMILES inputs. The model combines separate protein and ligand encoders, fuses a ligand fingerprint with a ligand sequence embedding, and relates the two branches through residue-level cross-attention.

# QSMpred Environment Setup for Local Deployment
QSMpred is validated for Windows. We recommend installation via conda. For set up using conda, please refer to the relevant code line provided below.
```
conda env create -f environment.yml
```

# Using of QSMpred
Using QSMpred includes both LBVS and SBVS parts. For the specific way to use the models, check the README.md in each folder.
