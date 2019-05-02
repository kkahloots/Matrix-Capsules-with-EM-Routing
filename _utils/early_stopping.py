"""
Created on Tue Sep 11 20:52:46 2018
@author: pablosanchez

early_stopping.py: deep learning training monitor, I left the old header tribute to the author
"""
__editor__      = "Khalid M. Kahloot"
__copyright__   = "Copyright 2019, Only for profesionals"

import _utils.constants as const

class EarlyStopping(object):
    def __init__(self, name='', patience=15, min_delta = const.tol):
        self.name = name
        self.patience = patience
        self.min_delta = min_delta
        self.patience_cnt = 0
        self.prev_loss_val = 200000000
        
    
    def stop(self, loss_val):
        if(self.prev_loss_val - loss_val> self.min_delta):
            self.patience_cnt = 0
            self.prev_loss_val = loss_val
            
        else:
            self.patience_cnt += 1
            print(self.name + ' Patience count: ', self.patience_cnt)
            
        if(self.patience_cnt > self.patience):
            return True
        else:
            return False
        
    