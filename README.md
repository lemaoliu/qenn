# Translation Quality Estimation from Bilingual Corpora 

It describes the main steps to train translation quality estimation systems only using bilingual corpus,
following our recent paper. 

## Train a machine translation system, for example, Moses [2], on one bilingual corpus. 

## Run the translation decoder to generate 1-best translations for the source side of another bilingual corpus

## Run the scripts to generate the tags for the 1-best translations using the target side (i.e. reference) of the above bilingual corpus.

Util now, we can get one dataset inlcuding: <source, (1-best) translation, tags>

## Run the word alignment between source and (1-best) translation using aligners such as fast align [3]


## Run the TQE trainer to learn the parameters for the model. 
We implemented translation quality estimation model based on feedforward neural networks following [4].
To train such TQE model using the generated dataset, please see the example in the fnn_tqe dir.
Before training, we have to make a config file as in fnn_tqe/config.ini

 


References:

