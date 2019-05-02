"""CapEM_model.py: 
A Tensorflow implementation of CapsNet based on paper Matrix Capsules with EM Routing
https://openreview.net/pdf?id=HJWLfGWRb
"""
__author__      = "Khalid M. Kahloot"
__copyright__   = "Copyright 2019, Only for profesionals"

import sys
sys.path.append('..')

from base.base_model import BaseModel
import tensorflow as tf
import numpy as np

from .CapEM_graph import CapEMGraph

from _utils.logger import Logger
from _utils.early_stopping import EarlyStopping
from tqdm import tqdm
import sys

import _utils.utils as utils
import _utils.constants as const

class CapEMModel(BaseModel):
    def __init__(self,network_params,
                 transfer_fct= tf.nn.relu,learning_rate=0.002,
                 kinit=tf.contrib.layers.xavier_initializer(),
                 regularizer=tf.contrib.layers.l2_regularizer(5e-04),
                 batch_size=32, 
                 epochs=200, checkpoint_dir='', 
                 summary_dir='', result_dir='', restore=0):
        super().__init__(checkpoint_dir, summary_dir, result_dir)
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.z_file = result_dir + '\\z_file'
    
        self.restore = restore
        #self.train = 0 if self.restore==1 else 1
        
        
        # Creating computational graph for train and test
        self.graph = tf.Graph()
        with self.graph.as_default():                
            self.model_graph = CapEMGraph(network_params, learning_rate, kinit, regularizer, batch_size, reuse=False)

            self.model_graph.build_graph()
            self.trainable_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
            
    
    def train_epoch(self, session,logger, data_train, m):
        loop = tqdm(range(data_train.num_batches(self.batch_size)))
        
        losses = []
        slosses = []
        reconsts = []
        L2_loss = []
        accs = []
        
        for _ in loop:
            batch_x, batch_y = next(data_train.next_batch(self.batch_size, with_labels=True))
            loss, spread_loss, reconst, L2_loss_curr, acc_curr  = self.model_graph.partial_fit(session, batch_x, batch_y, m)
            losses.append(loss)
            slosses.append(spread_loss)
            reconsts.append(reconst)
            L2_loss.append(L2_loss_curr)
            accs.append(acc_curr)

        loss_tr = np.mean(losses)
        sloss_tr = np.mean(slosses)
        recons_tr = np.mean(reconsts)
        L2_loss = np.mean(L2_loss)
        accs = np.mean(accs)
        
        cur_it = self.model_graph.global_step_tensor.eval(session)
        summaries_dict = {
            'loss': loss_tr,
            'spread_loss': sloss_tr,
            'reconsts': recons_tr,
            'L2_loss': L2_loss,
            'Acc': accs
        }
        
        logger.summarize(cur_it, summaries_dict=summaries_dict)
        
        return loss_tr, sloss_tr, recons_tr, L2_loss, accs
        
    def valid_epoch(self, session, logger, data_valid, m):
        # COMPUTE VALID LOSS
        loop = tqdm(range(data_valid.num_batches(self.batch_size)))
        
        losses_val = []
        slosses_val = []
        recons_val = []
        acc_val = []
        
        for _ in loop:
            batch_x, batch_y  = next(data_valid.next_batch(self.batch_size, with_labels=True))
            loss, sloss, recon, _, acc = self.model_graph.evaluate(session, batch_x, batch_y, m)
            
            losses_val.append(loss)
            slosses_val.append(sloss)
            recons_val.append(recon)
            acc_val.append(acc)
            
        loss_val = np.mean(losses_val)
        sloss_val = np.mean(slosses_val)
        recons_val = np.mean(recons_val)
        acc_val = np.mean(acc_val)

        cur_it = self.model_graph.global_step_tensor.eval(session)
        summaries_dict = {
            'loss': loss_val,
            'spread_loss': sloss_val,
            'recons_loss': recons_val,
            'acc_val': acc_val
        }
        logger.summarize(cur_it, summarizer="test", summaries_dict=summaries_dict)
        
        return loss_val, sloss_val, recons_val, acc_val
        
    def train(self, data_train, data_valid, enable_es=1):
        #Get batches per epoch
        num_batches_per_epoch = data_train.num_batches(self.batch_size)
        
        with tf.Session(graph=self.graph) as session:
            tf.set_random_seed(1234)
            
            logger = Logger(session, self.summary_dir)
            # here you initialize the tensorflow saver that will be used in saving the checkpoints.
            # max_to_keep: defaults to keeping the 5 most recent checkpoints of your model
            saver = tf.train.Saver()
            early_stopping = EarlyStopping()
            
            if(self.restore==1 and self.load(session, saver) ):
                num_epochs_trained = self.model_graph.cur_epoch_tensor.eval(session)
                print('EPOCHS trained: ', num_epochs_trained)      
            else:
                print('Initizalizing Variables ...')
                tf.global_variables_initializer().run()
                
                   
            if(self.model_graph.cur_epoch_tensor.eval(session) ==  self.epochs):
                return
                    
            #Main loop
            m_min = 0.2
            m_max = 0.9
            self.m = m_min
            for cur_epoch in range(self.model_graph.cur_epoch_tensor.eval(session), self.epochs + 1, 1):
        
                print('EPOCH: ', cur_epoch)
                self.current_epoch = cur_epoch
                
                
                loss_tr, sloss_tr, recons_tr, L2_loss, acc_tr = self.train_epoch(session, logger, data_train, self.m)
                if np.isnan(loss_tr):
                    print ('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                    print('spread_loss: ', sloss_tr)
                    print('Recons: ', recons_tr)
                    print('L2_loss: ', L2_loss)
                    print('Acc: ', acc_tr)
                    sys.exit()
                    
                loss_val, sloss_val, reconst_val, acc_val = self.valid_epoch(session, logger, data_valid, self.m)
                
                print('TRAIN | total Loss: ', loss_tr, ' | spread Loss: ', sloss_tr, ' | Recons: ', recons_tr, ' | L2_loss: ', L2_loss, ' | Acc: ', acc_tr)
                print('VALID | total Loss: ', loss_val, ' | spread Loss: ', sloss_val, ' | Recons: ', reconst_val, ' | Acc: ', acc_val)
                
                if(cur_epoch>0 and cur_epoch % 10 == 0):
                    self.save(session, saver, self.model_graph.global_step_tensor.eval(session))
                    
                session.run(self.model_graph.increment_cur_epoch_tensor)
                
                #Early stopping
                if(enable_es==1 and early_stopping.stop(loss_val)):
                    print('Early Stopping!')
                    break
                                
                                
                #Epoch wise linear annealling."""
                if (cur_epoch % num_batches_per_epoch) == 0:
                    if cur_epoch > 0:
                        self.m += (m_max - m_min) / (self.epochs * const.m_schedule)
                        if self.m > m_max:
                            self.m = m_max    
        
        return
    
            
    