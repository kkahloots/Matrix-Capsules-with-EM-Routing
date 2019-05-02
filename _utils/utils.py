"""utils.py: tensorflow util functions"""
__author__      = "Khalid M. Kahloot"
__copyright__   = "Copyright 2019, Only for profesionals"

from __future__ import division
import math
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
import matplotlib.pyplot as plt
import pandas as pd

import re
import os
import shutil
import argparse
import time
import gzip

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.preprocessing import MinMaxScaler
from skimage.color import grey2rgb

import sys
sys.path.append('..')

from _utils.dataset import Dataset

from scipy.io import loadmat
'''  ------------------------------------------------------------------------------
                                    DATA METHODS
 ------------------------------------------------------------------------------ '''
from sklearn.model_selection import train_test_split

scalar = None

def prepare_dataset(X):
    
    len_ = X.shape[0]
    shape_ = X.shape

    d = int(np.sqrt(X.flatten().reshape(X.shape[0],-1).shape[1]))

    if len(shape_)==4:
        X = np.reshape(X, [-1, d, d, 3])
        
    elif d==shape_[1] and len(shape_)==3:
        X = np.reshape(X, [-1, d, d])       
        X = np.array(list(map(lambda x: grey2rgb(x), X)), dtype=np.float32)

    else:
        r = d**2 - X.shape[1]
        train_padding = np.zeros((shape_[0], r))
        X = np.vstack([X, train_padding])
        
        X = np.reshape(X, [-1, d, d])
        X = np.array(list(map(lambda x: grey2rgb(x), X)), dtype=np.float32)
    
    print('Scaling dataset')
    if scalar is not None:
        X = scaler.transform(X.flatten().reshape(-1,1).astype(np.float32)).reshape(X.shape)
    else:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X.flatten().reshape(-1,1).astype(np.float32)).reshape(X.shape)
        
    return X
    
def process_data(X, y=None, test_size=0.20, dummies=False):
    if y is None:
        y = np.ones(X.shape[0])
    
    len_ = X.shape[0]    
    X = prepare_dataset(X)
        
    if dummies:
        y = pd.get_dummies(y).values
        
    shape_  = list(X.shape[1:])
    
    X_train, X_test, y_train, y_test  = train_test_split(X.flatten().reshape(len_,-1), y, test_size=test_size, random_state=4891)
    
    X_train = X_train.reshape([X_train.shape[0]]+shape_)
    X_test = X_test.reshape([X_test.shape[0]]+shape_)
     
    print('Training dataset shape: ', X_train.shape)
    print('Validation dataset shape: ', X_test.shape)

    train_dataset = Dataset(X_train, y_train)
    test_dataset = Dataset(X_test, y_test)    
    
    return train_dataset, test_dataset


def merge_datasets(data, data_dim, train_size, valid_size=0):
    valid_dataset = np.ndarray((valid_size, data_dim), dtype=np.float32)
    train_dataset = np.ndarray((train_size, data_dim), dtype=np.float32)

    np.random.shuffle(data)

    if valid_dataset is not None:
        valid_dataset = data[:valid_size, :]

    train_dataset = data[valid_size:, :]

    return valid_dataset, train_dataset

'''  ------------------------------------------------------------------------------
                                    FILES & DIRS
 ------------------------------------------------------------------------------ '''
 
class Config():
    def __init__(self, args):
        for k, v in args.items():
            setattr(self, k, v)
    
    def __getitem__(self, item):
         return self.__dict__[item]
         
'''  ------------------------------------------------------------------------------
                                    FILES & DIRS
 ------------------------------------------------------------------------------ '''
def get_model_name_CapEM(model, config):
    model_name = model + '_'+\
                 config.dataset_name+ '_' + \
                 str(config.cell_size).replace('.', '')+ '_'+\
                 str(config.hidden_caps) + '_'+ \
                 str(config.num_layers)
    return model_name
         
def save_img(fig, model_name, image_name, result_dir):
    complete_name = result_dir + '\\' + model_name + '_' + image_name + '.png'
    idx = 1
    while(os.path.exists(complete_name)):
        complete_name = result_dir + '\\' + model_name + '_' + image_name + '_'+str(idx)+'.png'
        idx+=1
    fig.savefig(complete_name)
    
def save_args(args, summary_dir):
    my_file = summary_dir + '\\' + 'my_args.txt'
    args_string = str(args).replace(', ', ' --')
    with open(my_file, 'a+') as file_:
        file_.write(args_string)
        
def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


'''  ------------------------------------------------------------------------------
                                    FOLDER/FILE METHODS
 ------------------------------------------------------------------------------ '''

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def clean_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=True)
    return

def open_log_file(filename, args):
    '''
    Open a file and writes the first line if it does not exists
    '''
    if(os.path.isfile(filename) ):
        return

    with open(filename, 'w+') as logfile:
        my_string = ''
        for arg in args[:-1]:
            my_string+= arg +';'

        my_string+= args[-1] + '\n'
        logfile.write(my_string)
    return

def write_log_file(filename, args):
    '''
    Write a line to a file with elements separated by commas.
    '''
    if(not os.path.isfile(filename) ):
        return

    with open(filename, 'a+') as logfile:
        my_string = ''
        for arg in args[:-1]:
            my_string+= arg +';'

        my_string+= args[-1] + '\n'
        logfile.write(my_string)
    return



'''  ------------------------------------------------------------------------------
                                    PRINT METHODS
 ------------------------------------------------------------------------------ '''

def printt(string, log):
    if(log):
        print(string)

def get_time():
    return strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + '\n'


def get_params(args):

    retval = ''
    for key in args:
        retval += '\t' + str(key) + ':' + str(args[key]) + '\n'
    return retval



'''  ------------------------------------------------------------------------------
                                    TF METHODS
 ------------------------------------------------------------------------------ '''

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def get_variable(dim, name, init_value=0.54):
    out = tf.get_variable(name,
                           initializer=tf.constant_initializer(init_value),
                           shape=[1, dim],
                           trainable=True,
                           dtype=tf.float32)
    out = tf.nn.softplus(out)
    return out

def variable_summary(var, name='summaries'):
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        
        # tf.summary.scalar('max', tf.reduce_max(var))
        # tf.summary.scalar('min', tf.reduce_min(var))
        
        tf.summary.histogram('histogram', var)
    return

def softplus_bias(tensor):
    out = tf.add(tf.nn.softplus(tensor), 0.1)
    return out

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# TensorFlow Graph visualizer code
import numpy as np
from IPython.display import clear_output, Image, display, HTML

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script src="//cdnjs.cloudflare.com/ajax/libs/polymer/0.3.3/platform.js"></script>
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))