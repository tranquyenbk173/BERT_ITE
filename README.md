# BERT-ITE
##### From Implicit to Explicit Feedback: A deep neural network for modeling sequential behaviours and long-short term preferences of online users

-----------------------------------------------
 This is our official implementation of BERT-ITE - a deep neural network for modeling sequential behaviours and long-short term preferences of online users. For more details, please refer to our papers: https://arxiv.org/abs/2107.12325

<p align="center">
<img align="center" src="https://github.com/tranquyenbk173/ITE/blob/master/src/result/image/MD1-1.png" width=100%>
</p>

-------------------------------
## Abstract:
In this work, we examine the advantages of using multiple types of behaviour in recommendation systems. Intuitively, each user has to do some implicit actions (e.g., click) before making an explicit decision (e.g., purchase). Previous studies showed that implicit and explicit feedback have differ-ent roles for a useful recommendation. However, these studies either exploit implicit and explicit behaviour separately or ignore the semantic of sequential interactions between users and items. In addition, we go from the hypothesis that a user’s preference at a time is a combination of long-term and short-term interests. In this paper, we propose some Deep Learning architectures. The first one isImplicit to Explicit (ITE), to exploit users’ interests through the sequence of their actions. And two versions of ITE with Bidirectional Encoder Representations from Transformers based (BERT-based) architecture called BERT-ITE andBERT-ITE-Si, which combine users’ long- and short-term preferences without and with side information to enhance user representation. The experimental results show that our models outperform previous state-of-the-art ones and also demonstrate our views on the effectiveness of exploiting the implicit to explicit order as well as combining long- and short-term preferences in two large scale datasets.

## Some experimental results:

<p align="center">
<img align="center" src="https://github.com/tranquyenbk173/ITE/blob/master/src/result/image/retail_epoch_main-1.png" width=100%>
 
</p>

<!-- <p align="center">
<img align="center" src="https://github.com/tranquyenbk173/ITE/blob/master/src/result/image/retail_recobell_numfactor_main-1.png" width=90% height=100%>
</p> -->

<p align="center">
<img align="center" src="https://github.com/tranquyenbk173/ITE/blob/master/src/result/image/multiTop-1.png" width=85%>
</p>


## How to run:

#### To run BERT-ITE's code:

```
$ cd src/model_bertIte_...
 
$ python3 main.py --num_factor 64\
                  --eta 0.1\
                  --lr 0.002\
                  --dataset tmall\
                  --batch_size 512
```
You can adjust the arguments.

#### To run ITE's code:
```
$ cd src/model_ite_...

$ python3 run_num_factor.py

or python3 run_batch_size.py

or python3 run_eta.py
```

## Citation

```
@article{TRAN202289,
title = {From implicit to explicit feedback: A deep neural network for modeling sequential behaviours and long-short term preferences of online users},
journal = {Neurocomputing},
volume = {479},
pages = {89-105},
year = {2022},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2022.01.023},
url = {https://www.sciencedirect.com/science/article/pii/S0925231222000418},
author = {Quyen Tran and Lam Tran and Linh Chu Hai and Ngo Van Linh and Khoat Than},
keywords = {Deep learning, Recommendation systems, Collaborative filtering, Implicit feedback, Explicit feedback, Long-term preference, Short-term preference},
}
```
