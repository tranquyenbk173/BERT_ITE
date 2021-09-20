# ITE and BERT-ITE
##### From Implicit to Explicit Feedback: A deep neural network for modeling sequential behaviours and long-short term preferences of online users

-----------------------------------------------
 This is our official implementation of ITE and BERT-ITE -  deep neural networks for modeling sequential behaviours and long-short term preferences of online users. For more details, please refer to our papers: (links)

<p align="center">
    <br>
    <a href="https://github.com/tranquyenbk173/src/result/image/MD1.pdf">
        <img src="https://github.com/tranquyenbk173/src/result/image/MD1.pdf" width="800"/>
    </a>
    <br>
<p>

-------------------------------
## Abstract:
In this work, we examine the advantages of using multiple types of behaviour in recommendation systems. Intuitively, each user has to do some implicit actions (e.g., click) before making an explicit decision (e.g., purchase). Previous studies showed that implicit and explicit feedback have differ-ent roles for a useful recommendation. However, these studies either exploit implicit and explicit behaviour separately or ignore the semantic of sequential interactions between users and items. In addition, we go from the hypothesis that a user’s preference at a time is a combination of long-term and short-term interests. In this paper, we propose some Deep Learning architectures. The first one isImplicit to Explicit (ITE), to exploit users’ interests through the sequence of their actions. And two versions of ITE with Bidirectional Encoder Representations from Transformers based (BERT-based) architecture called BERT-ITE andBERT-ITE-Si, which combine users’ long- and short-term preferences without and with side information to enhance user representation. The experimental results show that our models outperform previous state-of-the-art ones and also demonstrate our views on the effectiveness of exploiting the implicit to explicit order as well as combining long- and short-term preferences in two large scale datasets.

## Some experimental results:

<object data="https://.github.com/tranquyenbk173/src/result/image/retail_epoch_main.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://.github.com/tranquyenbk173/src/result/image/retail_epoch_main.pdf">

        <p> This browser does not support PDFs. Please download the PDF to view it: <a href="https://yoursite.com/the.pdf">Download PDF</a>.</p>

    </embed>
</object>

<object data="https://.github.com/tranquyenbk173/src/result/image/retail_recobell_numfactor_main.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://.github.com/tranquyenbk173/src/result/image/retail_recobell_numfactor_main.pdf">

        <p> This browser does not support PDFs. Please download the PDF to view it: <a href="https://yoursite.com/the.pdf">Download PDF</a>.</p>

    </embed>
</object>

<object data="https://.github.com/tranquyenbk173/src/result/image/multiTop.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://.github.com/tranquyenbk173/src/result/image/multiTop.pdf">

        <p> This browser does not support PDFs. Please download the PDF to view it: <a href="https://yoursite.com/the.pdf">Download PDF</a>.</p>

    </embed>
</object>

## How to run:

#### To run BERT-ITE code:

```
$ cd model_bertIte_...
 
$ python3 main.py --num_factor 64\
                --eta 0.1\
                --lr 0.002\
                --dataset tmall\
                --batch_size 512
```
You can adjust the parameters.

#### To run ITE code:
```
$ cd model_ite_...

$ python3 run_num_factor.py

or python3 run_batch_size.py

or python3 run_eta.py
```