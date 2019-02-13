# HeGAN
Source code for paper "Adversarial Learning on Heterogeneous Information Network"

## Evironment Setting

* Python == 2.7.3

* Tensorflow == 1.12.0

* Numpy == 1.15.1

## Parameter Setting (see config.py)
batch_size : The size of batch.

lambda_gen, lambda_dis : The regularization for generator and discriminator, respectively.

lr_gen, lr_dis : The learning rate for generator and discriminator, respectively.

n_epoch : The maximum training epoch.

sig : The variance of gaussian distribution in generator. 

g_epoch, d_epoch: The number of generator and discriminator training per epoch.

n_sample : The size of sample

n_emb : The embedding size

## Files in the folder

* data/: The training data

* result/: The learned embeddings of generator ane discriminator.

* code/: The source codes

* pre_train/: The pre-trained node embeddings (Note: The dimension of pre-trained node embeddings should equal n_emb)

## Data 
We privode three datasets: [DBLP](https://github.com/librahu/Heterogeneous-Information-Network-Datasets-for-Recommendation-and-Network-Embedding/tree/master/DBLP), [Yelp](https://github.com/librahu/Heterogeneous-Information-Network-Datasets-for-Recommendation-and-Network-Embedding/tree/master/Yelp_2) and [Aminer](https://github.com/librahu/Heterogeneous-Information-Network-Datasets-for-Recommendation-and-Network-Embedding/tree/master/Aminer), The detailed description of the three datasets can refer to https://github.com/librahu/Heterogeneous-Information-Network-Datasets-for-Recommendation-and-Network-Embedding

### The format of input training data
* Each line: source_node target_node relation

### The format of input pre-trained data
* The first line: node_num embedding_dim

* Each line : node_id embdeeing_1 embedding_2, ...

### The format of output embedding
* The first line: node_num embedding_dim

* Each line : node_id embdeeing_1 embedding_2, ...


## Basic Usage 

cd code

python he_gan.py


# Reference

@inproceedings{

> author = {Binbin Hu, Yuan Fang and Chuan Shi.},
 
> title = {Adversarial Learning on Heterogeneous Information Network},
 
<!--
> booktitle = {Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
 
> year = {2018},
 
> url = {https://dl.acm.org/citation.cfm?id=3219965},
 
> publisher = {ACM},

> address = {London, United Kingdom},
-->
> keywords = {Heterogeneous Information Network, Network Embedding, Generative Adversarial Network},
 
}
