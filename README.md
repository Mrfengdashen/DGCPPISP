# DGCPPISP
# Abstract
  We propose a PPI site prediction model (DGCPPISP) based on a dynamic graph convolutional neural network and a two-stage transfer learning strategy.First, we introduce the transfer learning from two aspects, which refer to the feature input and model training. In terms of the feature input, the feature embedding of the protein pre-training model ESM-2 is introduced as input to make up for the lack of conventional sequence feature information. For the model training, we use the protein-peptide binding residue dataset to pre-train the model so that the model has better initial parameters, and then transfer these weights to the target task of PPI site prediction for fine-tuning. Secondly, we construct the dynamic graph convolutional neural network to build a model for the second stage of training. Through the dynamic graph structure of dynamic graph convolution, we solve the problem of insufficient feature extraction caused by the fixed neighborhood of the graph neural network. Finally, we test the performance of DGCPPISP on two benchmark datasets, and the results show that DGCPPISP achieves the best results compared to other competitive methods. 
## Dataset
  Three datasets are used in this project, including Dataset_trans for transfer learning, Dataset 1 and Dataset 2 for model training and evaluation.All datasets can be obtained through the following path:
  * https://pan.baidu.com/s/1yvOgmKX1jCTMFpfZomVUOA?pwd=zjnb (Extraction code: zjnb)
## Using DGCPPISP
  You can train DGCPPISP and predict PPI sites through the dataset we provide.
### Model train
  ```python train_PPI.py```
### Model predict
  ```python predict_PPI.py```
