"""dataset.py: process, split, batch prepare for train deep learning"""
__author__      = "Khalid M. Kahloot"
__copyright__   = "Copyright 2019, Only for profesionals"

import numpy as np

class Dataset:
    def __init__(self, data, labels):
        self._x = data
        self._labels = labels
        
        self._ndata = data.shape[0]
        self.height = data.shape[2]
        self.width = data.shape[1]
        self.num_channels = data.shape[3]
        self._idx_batch = 0
        self._idx_vector = np.array(range(self._ndata))       

    @property
    def x(self):
        return self._x

    @property
    def labels(self):
        return self._labels
    
    def set_flat(self, flat):
        if(not flat):
            self._x = np.reshape(self._x, [-1, self.height, self.width, self._num_channels])
        else:
            self._x = np.reshape(
                self._x, [-1, self.height * self.width * self._num_channels])


   
    def num_batches(self, batch_size):
        return self._ndata//batch_size

    
    def shuffle(self, revert=False):
        if(revert == False):
            self._idx_vector = np.random.permutation(range(self._ndata))
        else:
            self._idx_vector = np.array(range(self._ndata))
    
    def shuffle_data(self, idx):
        self._x = self._x[idx]
        self._labels = self._labels[idx]
    
    def next_batch(self, batch_size, shuffle=True, with_labels=False):
        start = self._idx_batch
        if start == 0:
            if(shuffle):
                idx = np.arange(0, self._ndata)  # get all possible indexes
                np.random.shuffle(idx)  # shuffle indexe
                self.shuffle_data(idx)
        # go to the next batch
        if start + batch_size > self._ndata:
            rest_ndata = self._ndata - start
            
            x_rest_part = self._x[start:self._ndata]
            labels_rest_part = self._labels[start:self._ndata]
            
            if(shuffle):
                idx0 = np.arange(0, self._ndata)  # get all possible indexes
                np.random.shuffle(idx0)  # shuffle indexes
                self.shuffle_data(idx0)

            start = 0
            # avoid the case where the #sample != integar times of batch_size
            self._idx_batch = batch_size - rest_ndata
            end = self._idx_batch
            
            x_new_part = self._x[start:end]
            labels_new_part = self._labels[start:end]
            
            if not with_labels:
                yield np.concatenate((x_rest_part, x_new_part), axis=0) 
            else:
                yield np.concatenate((x_rest_part, x_new_part), axis=0),\
                      np.concatenate((labels_rest_part, labels_new_part), axis=0)
            
        else:
            self._idx_batch += batch_size
            end = self._idx_batch
            
        if not with_labels:
            yield self._x[start:end]
        else:
            yield self._x[start:end], self._labels[start:end]
       
        
    def take_n(self, n):
        self._ndata = n
        self._x = self._x[0:self._ndata]
        self._labels = self._labels[0:n]
        self._idx_vector = np.array(range(self._ndata))

            
    def random_batch(self, batch_size):
        idx = np.arange(0, self._ndata) 
        np.random.shuffle(idx)
        
        return self._x[idx[:batch_size]]
    
    def random_batch_with_labels(self, batch_size):
        idx = np.arange(0, self._ndata) 
        np.random.shuffle(idx)
        
        return self._x[idx[:batch_size]], self._labels[idx[:batch_size]]