# qenn

It describes the main steps to train translation quality estimation systems only using bilingual corpus,
following our recent paper. 

# train a machine translation system, for example, Moses, on one bilingual corpus.

# run the translation decoder to generate 1-best translations for the source side of another bilingual corpus

# run the scripts to generate the tags for the 1-best translations using the target side (i.e. reference) of the above bilingual corpus.

Util now, we can get one dataset inlcuding: <source, (1-best) translation, tags>

# run the TQE trainer to learn the parameters for the model. 
We implemented translation quality estimation model based on feedforward neural networks following [2].
To train such TQE model using the generated dataset, please see the example in the fnn_tqe dir.

 
