"""CapEM.py: actual im CapsuleEM build"""
__author__      = "Khalid M. Kahloot"
__copyright__   = "Copyright 2019, Only for profesionals"

import os
import tensorflow as tf
import numpy as np
import sys
sys.path.append('..')

from .CapEM_BASE import CapEM_BASE
import _utils.utils as utils
import _utils.constants as const

class CapEM(CapEM_BASE):
    def __init__(self, *argz, **kwrds):
        super(CapEM, self).__init__(*argz, **kwrds)
        self.config.model_name = 'CapEM'
        self.setup_logging()
        
    def _build(self):
       
        '''  ---------------------------------------------------------------------
                            COMPUTATION GRAPH (Build the model)
        ---------------------------------------------------------------------- '''
        from Alg_CapEM.CapEM_model import CapEMModel
        self.model = CapEMModel(self.network_params,
                               transfer_fct=tf.nn.relu,learning_rate=self.config.l_rate,
                               kinit=tf.contrib.layers.xavier_initializer(),
                               batch_size=self.config.batch_size,  
                               epochs=self.config.epochs, checkpoint_dir=self.config.checkpoint_dir, 
                               summary_dir=self.config.summary_dir, result_dir=self.config.results_dir, 
                               restore=self.flags.restore)
        print('building CapEM Model...')
        print('\nNumber of trainable paramters', self.model.trainable_count)
        
