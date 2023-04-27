#-*- encoding:utf8 -*-
import pickle

import numpy
import torch
import numpy as np
from torch.utils import data

#my lib
from config_PPI import DefaultConfig

class dataSet(data.Dataset):
    def __init__(self,window_size,sequences_file=None, label_file=None, protein_list_file=None):
        super(dataSet,self).__init__()
        
        self.all_sequences = []
        for seq_file in sequences_file:
            with open(seq_file,"rb") as fp_seq:
                temp_seq  = pickle.load(fp_seq)
            self.all_sequences.extend(temp_seq)

        self.all_label = []
        for lab_file in label_file: 
            with open(lab_file, "rb") as fp_label:
                temp_label = pickle.load(fp_label)
            self.all_label.extend(temp_label)

        with open(protein_list_file, "rb") as list_label:
            self.protein_list = pickle.load(list_label)

        self.Config = DefaultConfig()
        self.max_seq_len = self.Config.max_sequence_length
        self.window_size = window_size
        self.esm2_path = self.Config.esm2_path
        self.dipole_path = self.Config.dipole_path
        self.gram_path = self.Config.gram_path
        self.position_path = self.Config.position_path


    def __getitem__(self,index):
        count,id_idx,ii,dset,protein_id,seq_length = self.protein_list[index]
        all_esm2_features = np.load(self.esm2_path + dset + '/' + protein_id + '.npy',allow_pickle=True)
        all_dipole_features = np.load(self.dipole_path + dset + '/' + protein_id + '.npy', allow_pickle=True)
        all_gram_features = np.load(self.gram_path + dset + '/' + protein_id + '.npy', allow_pickle=True)
        all_position_features = np.load(self.position_path + dset + '/' + protein_id + '.npy', allow_pickle=True)
        all_esm2_features = all_esm2_features[:self.max_seq_len]
        all_dipole_features = all_dipole_features[:self.max_seq_len]
        all_gram_features = all_gram_features[:self.max_seq_len]
        all_position_features = all_position_features[:self.max_seq_len]

        window_size = self.window_size
        id_idx = int(id_idx)
        win_start = ii - window_size
        win_end = ii + window_size
        seq_length = int(seq_length)
        label_idx = (win_start+win_end)//2
        all_seq_features = []
        seq_len = 0
        for idx in self.all_sequences[id_idx][:self.max_seq_len]:
            acid_one_hot = [0 for i in range(21)]
            acid_one_hot[idx] = 1
            all_seq_features.append(acid_one_hot)
            seq_len += 1
        while seq_len<self.max_seq_len:
            acid_one_hot = [0 for i in range(21)]
            all_seq_features.append(acid_one_hot)
            seq_len += 1

        sub = 500 - all_esm2_features.shape[0]
        for i in range(0,sub):
            zero_vector = numpy.zeros((1,1280))
            all_esm2_features = np.append(all_esm2_features,zero_vector,axis=0)

        sub = 500 - all_dipole_features.shape[0]
        for i in range(0,sub):
            zero_vector = numpy.zeros((1,8))
            all_dipole_features = np.append(all_dipole_features,zero_vector,axis=0)

        sub = 500 - all_gram_features.shape[0]
        for i in range(0,sub):
            zero_vector = numpy.zeros((1,5))
            all_gram_features = np.append(all_gram_features,zero_vector,axis=0)

        sub = 500 - all_position_features.shape[0]
        for i in range(0,sub):
            zero_vector = numpy.zeros((1,20))
            all_position_features = np.append(all_position_features,zero_vector,axis=0)


        label = self.all_label[id_idx][label_idx]
        label = np.array(label,dtype=np.float32)

        all_seq_features = np.stack(all_seq_features)
        all_seq_features = all_seq_features[np.newaxis,:,:]

        # local_features = np.stack(local_features)
        all_esm2_features = all_esm2_features[np.newaxis,:,:]
        all_dipole_features = all_dipole_features[np.newaxis, :, :]
        all_gram_features = all_gram_features[np.newaxis, :, :]
        all_position_features = all_position_features[np.newaxis, :, :]

        label_idx = np.array([label_idx])
        all_seq_features = torch.from_numpy(all_seq_features)
        all_esm2_features = torch.Tensor(all_esm2_features)
        all_dipole_features = torch.Tensor(all_dipole_features)
        all_gram_features = torch.Tensor(all_gram_features)
        all_position_features = torch.Tensor(all_position_features)
        # all_pcp_features = torch.Tensor(all_pcp_features)
        label = torch.from_numpy(label)
        label_idx = torch.from_numpy(label_idx)

        return all_seq_features, all_esm2_features,  all_dipole_features,all_gram_features, all_position_features, label,label_idx

    def __len__(self):
        return len(self.protein_list)

if __name__ == '__main__':
    train_data = ["dset186", "dset164", "dset72"]
    train_sequences_file = ['PPI_data/data_cache/{0}_sequence_data.pkl'.format(key) for key in train_data]
    train_label_file = ['PPI_data/data_cache/{0}_label.pkl'.format(key) for key in train_data]
    all_list_file = 'PPI_data/data_cache/all_dset_list.pkl'
    train_list_file = 'PPI_data/data_cache/training_list.pkl'
    train = dataSet(3,train_sequences_file,train_label_file,all_list_file)
    i=0
    for decoy in train:
        i+=1
        print(i)
