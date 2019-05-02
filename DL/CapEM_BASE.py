"""CapEM_BASE.py: base class for CapsuleEM, build, fit, predict methods"""
__author__      = "Khalid M. Kahloot"
__copyright__   = "Copyright 2019, Only for profesionals"

import os
import tensorflow as tf
import numpy as np
import sys
sys.path.append('..')

import _utils.utils as utils
from _utils.utils import prepare_dataset, get_model_name_CapEM

class CapEM_BASE():
    '''  ------------------------------------------------------------------------------
                                         SET ARGUMENTS
        ---------------------------------------------------------------------------------- '''
    
    def __init__(self, dataset_name, cell_size=8.0, hidden_caps=4, num_layers=3, epochs=100, batch_size=32, l_rate=1e-05, restore=0):
        args=dict()
        args['model_type']=0
        args['model_name']='CapEM'
        args['dataset_name']=dataset_name
        
        args['cell_size']=cell_size
        args['hidden_caps']=hidden_caps
        args['num_layers']=num_layers
               
        args['epochs']=epochs
        args['batch_size']=batch_size
        args['l_rate']=l_rate
        args['train']=0 if restore==1 else 1
        args['results']=1
        args['plot']=1
        args['restore']=restore
        args['early_stopping']=1
        
        coords = list()
        d=cell_size
        for i, l in enumerate(range(num_layers)):
            capLayer = list()
            for j, c in enumerate(range(hidden_caps)):
                if j==0:
                    capLayer.append([d, (d+j*d+i*d)//2])
                else:
                    capLayer.append([(d+j*d+i*d)//2, (d+i*d)//2])
            coords.append(capLayer)
        args['coords']=coords    
        
        
        dirs = ['checkpoint_dir','summary_dir', 'result_dir', 'log_dir' ]
        for dr in dirs:
            args[dr] = dr
        
        self.config = utils.Config(args)
        
    def setup_logging(self):        
        experiments_root_dir = 'experiments'
        self.config.model_name = get_model_name_CapEM(self.config.model_name, self.config)
        self.config.summary_dir = os.path.join(experiments_root_dir+"\\"+self.config.log_dir+"\\", self.config.model_name)
        self.config.checkpoint_dir = os.path.join(experiments_root_dir+"\\"+self.config.checkpoint_dir+"\\", self.config.model_name)
        self.config.results_dir = os.path.join(experiments_root_dir+"\\"+self.config.result_dir+"\\", self.config.model_name)

        #Flags
        flags_list = ['train', 'restore', 'results', 'plot', 'early_stopping']
        self.flags = utils.Config({ your_key: self.config.__dict__[your_key] for your_key in flags_list})
        
        # create the experiments dirs
        utils.create_dirs([self.config.summary_dir, self.config.checkpoint_dir, self.config.results_dir])
        utils.save_args(self.config.__dict__, self.config.summary_dir)
 
    def fit(self, X, y=None):
        '''  ------------------------------------------------------------------------------
                                         DATA PROCESSING
        ------------------------------------------------------------------------------ '''
            
        print('\n Processing data...')
        self.data_train, self.data_valid = utils.process_data(X, y, dummies=True)

        print('\n building a model...')
        self.build()
        
        '''  -------------------------------------------------------------------------------
                        TRAIN THE MODEL
        ------------------------------------------------------------------------------------- '''

        if(self.flags.train==1):
            print('\n training a model...')
            self.model.train(self.data_train, self.data_valid, enable_es=self.flags.early_stopping)

    
    def predict(self, X):
        
        with tf.Session(graph=self.model.graph) as session:
            saver = tf.train.Saver()
            if(self.model.load(session, saver)):
                num_epochs_trained = self.model.model_graph.cur_epoch_tensor.eval(session)
                print('EPOCHS trained: ', num_epochs_trained)
            else:
                return
                    
            #if len(X.shape)<4:
                #X = np.array([X])
            
            X_ = prepare_dataset(X)            
            y_l = list()
            
            start=0
            end= self.model.batch_size
            
            while end < X.shape[0]:
                x = X_[start:end]
                print('from {} to {}'.format(start, end))
                
                y_pred = self.model.model_graph.predict(session, x)

                y_l.append(np.array(y_pred[0]))

                start=end
                end +=self.model.batch_size
                
            else:
                    
                x = X_[start:]
                xsize = len(x)

                print('from {} to {}'.format(start, len(X_)))
                
                p = np.zeros([self.model.batch_size-xsize]+ list(x.shape[1:]))
                
                y_pred = self.model.model_graph.predict(session, np.concatenate((x,p), axis=0))

                y_l.append(np.array(y_pred[0][0:xsize]))
                
        return np.vstack(y_l)
    
    def build(self):
        '''  ------------------------------------------------------------------------------
                                     SET NETWORK PARAMS
        ------------------------------------------------------------------------------ '''        
        network_params_dict = dict()
        network_params_dict['input_height'] = self.data_train.height
        network_params_dict['input_width'] = self.data_train.width
        network_params_dict['input_nchannels'] = self.data_train.num_channels
        network_params_dict['train_size'] = self.data_train._ndata
        network_params_dict['num_classes'] = self.data_train._labels.shape[1]
        
        network_params_dict['cell_size'] =  self.config.cell_size
        network_params_dict['hidden_caps'] =  self.config.hidden_caps
        network_params_dict['num_layers'] =  self.config.num_layers
        
        self.network_params = utils.Config(network_params_dict)  
        
        self._build()    
     
    def _build(self):
        pass

