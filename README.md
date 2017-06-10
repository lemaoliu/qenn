# Translation Quality Estimation (TQE) only Using Bilingual Corpora 

It describes the main steps to train translation quality estimation systems only using bilingual corpus,
following our recent paper. 

## Train a machine translation system on one bilingual corpus, and then generate 1-best translations for the source side of another bilingual corpus. 
We used Moses [2] as the translation system. For setup of Mosese, please follow the guideline from 
http://www.statmt.org/moses.


## Run the scripts to generate the tags for the 1-best translations using the target side (i.e. reference) of the above bilingual corpus.



## Run the word alignment between source and (1-best) translation using aligners such as fast align [3]

Now we can get one dataset inlcuding <source, (1-best) translation, tags, alignment> for training.

## Run the TQE trainer to learn the parameters for the model. 
We implemented translation quality estimation model based on feedforward neural networks following [4].
To train such TQE model using the generated dataset, please see the example in the fnn_tqe dir.
Before training, we have to make a config file as in fnn_tqe/config.ini.
``
In fnn_tqe/config.ini, each line includes 3 fields as options (type, key, value). For example,
"int batch_size  400" indicates that the option batch_size is with value of an integer 400, and similarly,
"str trg_file train.target" indicates that the option trg_file is with value of a string train.target.
Additionally, these values of options denote the path of files: 
1. train.source, the source side of TQE training data
2. train.target, the (1-best) translation of train.source
3. train.tags, the tag file of train.target
4. train.align, the alignment file between train.source and train.target
5. dev.source, the source of valid file.

...


For training, just run the cmd:

``
python nn_bidir.py
`` 

## References:
1. Liu et al., Translation Quality Estimation Using Only Bilingual Corpora, 
IEEE/ACM Transactions on Audio, Speech and Language Processing
(TASLP), vol. xx, no. xx, pp.xx-xx, 2017.

2. Koehn et al., Moses: Open source toolkit for statistical machine trans- lation, in Proceedings of the 45th Annual Meeting of the Association for Computational Linguistics Companion Volume Proceedings of the Demo and Poster Sessions, 2007. 

3. Dyer et al., A simple, fast, and effective reparameterization of IBM model 2,
 in Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2013.

4. Kreutzer et al., Quality estimation from scratch (QUETCH): Deep learning for word-level translation 
quality estimation, in Proceedings of the Tenth Workshop on Statistical Machine Translation, 2015. 


