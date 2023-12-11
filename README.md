# 《ConeE: Global and Local Context-enhanced Embedding for Inductive Knowledge Graph Completion》  
Authors: Jingchao Wang, Weimin Li, etc

Abstract:

Knowledge graph completion (KGC) aims at completing missing information in knowledge graphs (KGs). Most previous works work well in the transductive setting, but are not applicable in the inductive setting, i.e., test entities can be unseen during training. Recently proposed methods obtain inductive ability by learning logic rules from subgraphs. However, all these works only consider the structural information of subgraphs while ignoring the rich contextual semantic information underlying KGs, which tends to lead to a sub-optimal embedding result. Furthermore, they tend to perform poorly when the subgraphs are sparse. To address these problems, we propose a global and local \textbf{Con}text-\textbf{e}nhanced \textbf{E}mbedding network, ConeE, which can fully utilize local and global contextual information to enhance embedding representations through the following two components. (1) The global context modeling module (GCMM) is a semi-parametric coarse-grained global semantic extractor, which can effectively extract global context-based semantic information via a BERT-based context encoder and a semantic fusion network (SFN), and adopts a novel contrastive learning-based sampling strategy to optimize semantic features. Furthermore, a scoring network is designed to evaluate the confidence of triplets from the perspective of both the triplet facts and the reasoning path to improve the accuracy of prediction. (2) The local context modeling module (LCMM) employs an interactive graph neural network (IGNN) to extract local topological features from subgraphs, and applies mutual information maximization (MIM) to subgraph modeling to capture more local features. Experiments on benchmark datasets show that ConeE significantly outperforms existing state-of-the-art methods. 

## Dependencies
The code is based on Python 3.7. In addition, you need to add the following dependencies to your environment：
    dgl==0.4.2
    lmdb==0.98
    networkx==2.4
    scikit-learn==0.22.1
    torch==1.4.0
    tqdm==4.43.0

## Code structure

In the root directory, the “Data” folder contains all the data sets used in this work, and the “Code” folder contains all the codes.

### How to train
Please change to the Code directory, and then use the following command to train the model.

Train WN18RR dataset using the following commands:

```shell script
python train.py -d WN18RR_v1 -e WN18RR_v1
python train.py -d WN18RR_v2 -e WN18RR_v2
python train.py -d WN18RR_v3 -e WN18RR_v3
python train.py -d WN18RR_v4 -e WN18RR_v4
```

Train Fb15K-237 dataset using the following commands:
```shell script
python train.py -d fb237_v1 -e Fb15K237_v1
python train.py -d fb237_v2 -e Fb15K237_v2
python train.py -d fb237_v3 -e Fb15K237_v3
python train.py -d fb237_v4 -e Fb15K237_v4
```
Train NELL-995 dataset using the following commands:
```shell script
python train.py -d fb237_v1 -e NELL995_v1
python train.py -d fb237_v2 -e NELL995_v2
python train.py -d fb237_v3 -e NELL995_v3
python train.py -d fb237_v4 -e NELL995_v4
```

### How to evaluate 

Taking WN18RR as an example, use a command similar to the following to evaluate the model:

```shell script
python test_auc.py -d WN18RR_v1_ind -e WN18RR_v1
python test_ranking.py -d WN18RR_v1_ind -e WN18RR_v1
```





## Acknowledgement
We refer to the code of [GraIL](https://github.com/kkteru/grail). Thanks for their contributions.

More details can be found in the code. If you have any questions please contact us.
