# DGCPPISP
  A PPI site prediction model based on dynamic graph convolutional network and two-stage transfer learning
## Abstract
  We propose a PPI site prediction model (DGCPPISP) based on a dynamic graph convolutional neural network and a two-stage transfer learning strategy.First, we introduce the transfer learning from two aspects, which refer to the feature input and model training. In terms of the feature input, the feature embedding of the protein pre-training model ESM-2 is introduced as input to make up for the lack of conventional sequence feature information. For the model training, we use the protein-peptide binding residue dataset to pre-train the model so that the model has better initial parameters, and then transfer these weights to the target task of PPI site prediction for fine-tuning. Secondly, we construct the dynamic graph convolutional neural network to build a model for the second stage of training. Through the dynamic graph structure of dynamic graph convolution, we solve the problem of insufficient feature extraction caused by the fixed neighborhood of the graph neural network. Finally, we test the performance of DGCPPISP on two benchmark datasets, and the results show that DGCPPISP achieves the best results compared to other competitive methods. 
## Datasets
  Three datasets are used in this project, including Dataset_trans for transfer learning, Dataset 1 and Dataset 2 for model training and evaluation. All these raw datasets can be obtained from [Datasets](https://github.com/CSUBioGroup/DeepPPISP),processed data from [Processed Datasets](https://drive.google.com/file/d/123x5dxfpJnoGoNRs-ARS5wed-seusGTh/view?usp=sharing) 
  The protein pre-trained language model ESM-2 is obtained from [here](https://github.com/facebookresearch/esm). The version used in this article can be downloaded from [ESM-2](https://drive.google.com/file/d/1lfOTtX7N-6hqFJQdAK5B78ko2uDLypcG/view?usp=drive_link).
## Using DGCPPISP
  You can train DGCPPISP and predict PPI sites through the dataset we provide.
### Model train
```
python train_PPI.py
```
### Model predict
```
python predict_PPI.py
```
