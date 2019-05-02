"""CapEM_graph.py: 
A Tensorflow implementation of CapsNet computational graph
"""

from base.base_graph import BaseGraph

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import _utils.constants as const

class CapEMGraph(BaseGraph):
    def __init__(self, network_params, learning_rate=0.002,
                 kinit=tf.contrib.layers.xavier_initializer(),
                 regularizer=tf.contrib.layers.l2_regularizer(5e-04),
                 batch_size=32, reuse=None):
        
        super().__init__(learning_rate)
        
        self.width = network_params['input_width']
        self.height = network_params['input_height']
        self.nchannel = network_params['input_nchannels']
        
        self.num_classes = network_params['num_classes']      
        
        self.height = network_params['input_height']
        self.width = network_params['input_width']
        self.nchannel = network_params['input_nchannels']
        self.train_size = network_params['train_size'] 
    
        self.cell_size = d = network_params['cell_size']  
        self.hidden_caps = network_params['hidden_caps']
        self.num_layers = network_params['num_layers']
        
        self.coords = list()
        
        for i, l in enumerate(range(self.num_layers)):
            capLayer = list()
            for j, c in enumerate(range(self.hidden_caps)):
                if j==0:
                    capLayer.append([d, (d+j*d+i*d)//2])
                else:
                    capLayer.append([(d+j*d+i*d)//2, (d+i*d)//2])
            self.coords.append(capLayer)
            
        self.coords = np.array(self.coords)/self.width
        self.coords = [[[8., 8.], [12., 8.], [16., 8.]],
                          [[8., 12.], [12., 12.], [16., 12.]],
                          [[8., 16.], [12., 16.], [16., 16.]]]
        self.coords = np.array(self.coords)/self.width                  
        
        self.x_flat_dim = self.width * self.height * self.nchannel
        
        self.kinit = kinit
        self.regularizer = regularizer   # The paper didnot mention any regularization, a common l2 regularizer to weights is added here
        
        self.bias_init = tf.constant_initializer(0.)
        self.batch_size = batch_size
        
        self.reuse = reuse

    def build_graph(self):
        self.create_inputs()
        self.create_graph()
        self.create_loss_optimizer()
    
    def create_inputs(self):
        with tf.variable_scope('inputs', reuse=self.reuse):
            self.x_batch = tf.placeholder(tf.float32, [self.batch_size, self.width, self.height,  self.nchannel], name='x_batch')
            self.y_batch = tf.placeholder(tf.float32, shape=[self.batch_size,self.num_classes], name='y_batch')
            self.m_op = tf.placeholder(dtype=tf.float32, shape=())
        
    def create_graph(self):
        print('\n[*] Defining Capsule Graph...')

        test1 = []
        data_size = int(self.x_batch.get_shape()[1])     

        print('input shape: {}'.format(self.x_batch.get_shape()))

        # weights_initializer=initializer,
        #with slim.arg_scope([tf.layers.conv3d], trainable=True, biases_initializer=self.bias_init, weights_regularizer=self.regularizer):
        with tf.variable_scope('relu_conv1') as scope:
            output = tf.layers.conv2d(self.x_batch, filters=const.convA, kernel_size=[5, 5], strides=2, padding='VALID', name=scope, activation=tf.nn.relu)
            data_size = int(np.floor((data_size - 4) / 2))

            assert output.get_shape() == [self.batch_size, data_size, data_size, const.convA]
            print('conv1 output shape: {}'.format(output.get_shape()))

        with tf.variable_scope('primary_caps') as scope:
            pose = tf.layers.conv2d(output, filters=const.capB * 16,
                               kernel_size=[1, 1], strides=1, padding='VALID', name=scope, activation=None)
            activation = tf.layers.conv2d(output, filters=const.capB, kernel_size=[1, 1], strides=1, padding='VALID',
                                                                     name='primary_caps/activation', activation=tf.nn.sigmoid)
            pose = tf.reshape(pose, shape=[self.batch_size, data_size, data_size, const.capB, 16])
            activation = tf.reshape(activation, shape=[self.batch_size, data_size, data_size, const.capB, 1])
                
            output = tf.concat([pose, activation], axis=4)
            output = tf.reshape(output, shape=[self.batch_size, data_size, data_size,-1])
            assert output.get_shape() == [self.batch_size, data_size, data_size, const.capB * 17]
            print('primary capsule output shape: {}'.format(output.get_shape()))

        with tf.variable_scope('conv_caps1') as scope:
            output = self.kernel_tile(output, 3, 2)
            data_size = int(np.floor((data_size - 2) / 2))
            output = tf.reshape(output, shape=[self.batch_size * data_size * data_size , 3 * 3 * const.capB, 17])
            activation = tf.reshape(output[:, :, 16], shape=[self.batch_size * data_size * data_size , 3 * 3 * const.capB, 1])

            with tf.variable_scope('v') as scope:
                votes = self.mat_transform(output[:, :, :16], const.capC, self.kinit, self.regularizer, tag=True)
                print('conv cap 1 votes shape: {}'.format(votes.get_shape()))

            with tf.variable_scope('routing') as scope:
                miu, activation = self.em_routing(votes, activation, const.capC, self.kinit, self.regularizer)
                print('conv cap 1 miu shape: {}'.format(miu.get_shape()))
                print('conv cap 1 activation before reshape: {}'.format(activation.get_shape()))

            pose = tf.reshape(miu, shape=[self.batch_size, data_size, data_size, const.capC, 16])
            print('conv cap 1 pose shape: {}'.format(pose.get_shape()))
            activation = tf.reshape(
                activation, shape=[self.batch_size, data_size, data_size, const.capC, 1])
            print('conv cap 1 activation after reshape: {}'.format(
                activation.get_shape()))
            output = tf.reshape(tf.concat([pose, activation], axis=4), [self.batch_size, data_size, data_size, -1])
            print('conv cap 1 output shape: {}'.format(output.get_shape()))

        with tf.variable_scope('conv_caps2') as scope:
            output = self.kernel_tile(output, 3, 1)
            data_size = int(np.floor((data_size - 2) / 1))
            output = tf.reshape(output, shape=[self.batch_size * data_size * data_size , 3 * 3 * const.capC, 17])
            activation = tf.reshape(output[:, :, 16], shape=[self.batch_size * data_size * data_size , 3 * 3 * const.capC, 1])

            with tf.variable_scope('v') as scope:
                votes = self.mat_transform(output[:, :, :16], const.capD, self.kinit, self.regularizer)
                print('conv cap 2 votes shape: {}'.format(votes.get_shape()))

            with tf.variable_scope('routing') as scope:
                miu, activation = self.em_routing(votes, activation, const.capD, self.kinit, self.regularizer)

            pose = tf.reshape(miu, shape=[self.batch_size * data_size * data_size , const.capD, 16])
            print('conv cap 2 pose shape: {}'.format(votes.get_shape()))
            activation = tf.reshape(
                activation, shape=[self.batch_size * data_size * data_size, const.capD, 1])
            print('conv cap 2 activation shape: {}'.format(activation.get_shape()))

        # It is not clear from the paper that ConvCaps2 is full connected to Class Capsules, or is conv connected with kernel size of 1*1 and a global average pooling.
        # From the description in Figure 1 of the paper and the amount of parameters (310k in the paper and 316,853 in fact), I assume a conv cap plus a golbal average pooling is the design.
        with tf.variable_scope('class_caps') as scope:
            with tf.variable_scope('v') as scope:
                votes = self.mat_transform(pose, self.num_classes, self.kinit, self.regularizer)

                assert votes.get_shape() == [self.batch_size * data_size * data_size, const.capD, self.num_classes, 16]
                print('class cap votes original shape: {}'.format(votes.get_shape()))

                coord_add = np.reshape(self.coords, newshape=[data_size * data_size , 1, 1, 2])
                coord_add = np.tile(coord_add, [self.batch_size, const.capD, self.num_classes, 1])
                coord_add_op = tf.constant(coord_add, dtype=tf.float32)

                votes = tf.concat([coord_add_op, votes], axis=3)
                print('class cap votes coord add shape: {}'.format(votes.get_shape()))

            with tf.variable_scope('routing') as scope:
                miu, activation = self.em_routing(votes, activation, self.num_classes, self.kinit, self.regularizer)
                print('class cap activation shape: {}'.format(activation.get_shape()))


            output = tf.reshape(activation, shape=[
                                self.batch_size, data_size, data_size, self.num_classes])

        self.y_pred = tf.reshape(tf.nn.avg_pool(output, ksize=[1, data_size, data_size, 1], strides=[1, 1, 1, 1], 
                                                        padding='VALID'), shape=[self.batch_size, self.num_classes])
        print('class cap output shape: {}'.format(self.y_pred.get_shape()))

        pose = tf.nn.avg_pool(tf.reshape(miu, shape=[self.batch_size, data_size, data_size, -1]), ksize=[1, data_size, data_size, 1],
                                                                                                 strides=[1, 1, 1, 1], padding='VALID')
        self.pose_out = tf.reshape(pose, shape=[self.batch_size, self.num_classes, 18])
        
        
    def create_loss_optimizer(self):
        print('[*] Defining Loss Functions and Optimizer...')
        num_class = int(self.y_pred.get_shape()[-1])
        data_size = int(self.x_batch.get_shape()[1])

        #y = tf.one_hot(y, num_class, dtype=tf.float32)

        # spread loss 
        output1 = tf.reshape(self.y_pred, shape=[self.batch_size, 1, num_class])
        y = tf.expand_dims(self.y_batch, axis=2)
        at = tf.matmul(output1, y)
        
        ##Paper eq(5)
        loss = tf.square(tf.maximum(0., self.m_op - (at - output1)))
        loss = tf.matmul(loss, 1. - y)
        
        with tf.variable_scope("spread_loss", reuse=self.reuse):
            self.spread_loss = tf.reduce_mean(loss)

        # reconstruction loss
        pose_out = tf.reshape(tf.multiply(self.pose_out, y), shape=[self.batch_size, -1])
        print("decoder input value dimension:{}".format(pose_out.get_shape()))

        with tf.variable_scope('reconstruct'):
            pose_out = slim.fully_connected(pose_out, 512, trainable=True, weights_regularizer=tf.contrib.layers.l2_regularizer(5e-04))
            pose_out = slim.fully_connected(pose_out, 1024, trainable=True, weights_regularizer=tf.contrib.layers.l2_regularizer(5e-04))
            pose_out = slim.fully_connected(pose_out, data_size * data_size * 3,
                                            trainable=True, activation_fn=tf.sigmoid, weights_regularizer=tf.contrib.layers.l2_regularizer(5e-04))

            #x = tf.reshape(self.x_batch, shape=[self.batch_size, -1])
            self.reconstruction_loss = tf.reduce_mean(tf.square(pose_out - tf.reshape(self.x_batch,shape=[self.x_batch.shape[0],-1])))

        
        self.loss_all = tf.add_n([self.spread_loss] + [0.0005 * data_size * data_size * 3 * self.reconstruction_loss])
        
        with tf.variable_scope("L2_loss", reuse=self.reuse):
            tv = tf.trainable_variables()
            self.L2 = tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])
        
        
        with tf.variable_scope("optimizer" ,reuse=self.reuse):
            #Get batches per epoch
            num_batches_per_epoch = int(self.train_size / self.batch_size)
            
            #Use exponential decay leanring rate?
            lrn_rate = tf.maximum(tf.train.exponential_decay(1e-3, self.global_step_tensor, num_batches_per_epoch, 0.8), 1e-5)       
            self.optimizer = tf .train.AdamOptimizer(lrn_rate)
            self.train_step = self.optimizer.minimize(self.loss_all, global_step=self.global_step_tensor)

    '''  ------------------------------------------------------------------------------
                                     EVALUATE TENSORS
    ------------------------------------------------------------------------------ '''
    def partial_fit(self, session, x, y, m):
        tensors = [self.train_step, self.loss_all, self.spread_loss, self.reconstruction_loss, self.L2, self.y_pred]
        feed_dict = {self.x_batch: x, self.y_batch: y, self.m_op: m }
        _, loss_all, spread_loss, reconst, L2_loss, y_pred = session.run(tensors, feed_dict=feed_dict)
        acc = self.test_accuracy(y_pred, y)
        acc = session.run(acc)
        return loss_all, spread_loss, reconst, L2_loss, acc
    
    def evaluate(self, session, x, y, m):
        tensors = [self.loss_all, self.spread_loss, self.reconstruction_loss, self.L2, self.y_pred]
        feed_dict = {self.x_batch: x, self.y_batch: y, self.m_op: m}
        loss_all, spread_loss, reconst, L2_loss, y_pred  = session.run(tensors, feed_dict=feed_dict)
        acc = self.test_accuracy(y_pred, y)
        acc = session.run(acc)
        return loss_all, spread_loss, reconst, L2_loss, acc
        
    def predict(self,session, x):
        tensors = [self.y_pred]
        feed_dict = {self.x_batch: x}
        y_pred  = session.run(tensors, feed_dict=feed_dict)
        return y_pred    

    '''  ------------------------------------------------------------------------------
                                     EM Routing
    ------------------------------------------------------------------------------ '''
    
        
    def em_routing(self, votes, activation, caps_num_c, initializer, regularizer, tag=False):

        batch_size = int(votes.get_shape()[0])
        caps_num_i = int(activation.get_shape()[1])
        n_channels = int(votes.get_shape()[-1])

        sigma_square = []
        miu = []
        activation_out = []
        beta_v = tf.Variable(name='beta_v', expected_shape=[caps_num_c, n_channels], dtype=tf.float32, initial_value=initializer([caps_num_c, n_channels]))
        beta_a = tf.Variable(name='beta_a', expected_shape=[caps_num_c], dtype=tf.float32, initial_value=initializer([caps_num_c]))

        votes_in = votes
        activation_in = activation

        for iters in range(const.iter_routing):
            # e-step
            if iters == 0:
                r = tf.constant(np.ones([batch_size, caps_num_i, caps_num_c], dtype=np.float32) / caps_num_c)
            else:
                # Contributor: Yunzhi Shi
                # log and exp here provide higher numerical stability especially for bigger number of iterations
                log_p_c_h = -tf.log(tf.sqrt(sigma_square)) - \
                            (tf.square(votes_in - miu) / (2 * sigma_square))
                log_p_c_h = log_p_c_h - \
                            (tf.reduce_max(log_p_c_h, axis=[2, 3], keep_dims=True) - tf.log(10.0))
                p_c = tf.exp(tf.reduce_sum(log_p_c_h, axis=3))

                ap = p_c * tf.reshape(activation_out, shape=[batch_size, 1, caps_num_c])
                r = ap / (tf.reduce_sum(ap, axis=2, keep_dims=True) + const.epsilon)

            # m-step
            r = r * activation_in
            r = r / (tf.reduce_sum(r, axis=2, keep_dims=True)+const.epsilon)

            r_sum = tf.reduce_sum(r, axis=1, keep_dims=True)
            r1 = tf.reshape(r / (r_sum + const.epsilon),
                            shape=[batch_size, caps_num_i, caps_num_c, 1])

            miu = tf.reduce_sum(votes_in * r1, axis=1, keep_dims=True)
            sigma_square = tf.reduce_sum(tf.square(votes_in - miu) * r1,
                                         axis=1, keep_dims=True) + const.epsilon

            if iters == const.iter_routing-1:
                r_sum = tf.reshape(r_sum, [batch_size, caps_num_c, 1])
                cost_h = (beta_v + tf.log(tf.sqrt(tf.reshape(sigma_square,
                                                             shape=[batch_size, caps_num_c, n_channels])))) * r_sum

                activation_out = tf.nn.softmax(const.ac_lambda0 * (beta_a - tf.reduce_sum(cost_h, axis=2)))
            else:
                activation_out = tf.nn.softmax(r_sum)

        return miu, activation_out        

    def kernel_tile(self, input, kernel, stride):
        # output = tf.extract_image_patches(input, ksizes=[1, kernel, kernel, 1], strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='VALID')

        input_shape = input.get_shape()
        tile_filter = np.zeros(shape=[kernel, kernel, input_shape[3],
                                      kernel * kernel], dtype=np.float32)
        for i in range(kernel):
            for j in range(kernel):
                tile_filter[i, j, :, i * kernel + j] = 1.0

        tile_filter_op = tf.constant(tile_filter, dtype=tf.float32)
        output = tf.nn.depthwise_conv2d(input, tile_filter_op, strides=[1, stride, stride, 1], padding='VALID')
        output_shape = output.get_shape()
        output = tf.reshape(output, shape=[int(output_shape[0]), int(
            output_shape[1]), int(output_shape[2]), int(input_shape[3]), kernel * kernel])
        output = tf.transpose(output, perm=[0, 1, 2, 4, 3])

        return output    
    
    # input should be a tensor with size as [batch_size, caps_num_i, 16]
    def mat_transform(self, input, caps_num_c, initializer, regularizer, tag=False):
        batch_size = int(input.get_shape()[0])
        caps_num_i = int(input.get_shape()[1])
        output = tf.reshape(input, shape=[batch_size, caps_num_i, 1, 4, 4])
        # the output of capsule is miu, the mean of a Gaussian, and activation, the sum of probabilities
        # it has no relationship with the absolute values of w and votes
        # using weights with bigger stddev helps numerical stability
        shape_ = [1, caps_num_i, caps_num_c, 4, 4]
        w = tf.Variable(name='w', expected_shape=shape_, dtype=tf.float32, initial_value=initializer(shape_))

        w = tf.tile(w, [batch_size, 1, 1, 1, 1])
        output = tf.tile(output, [1, 1, caps_num_c, 1, 1])
        votes = tf.reshape(tf.matmul(output, w), [batch_size, caps_num_i, caps_num_c, 16])

        return votes    
        
        
    def test_accuracy(self, logits, labels):
        logits_idx = tf.to_int32(tf.argmax(logits, axis=1))
        logits_idx = tf.reshape(logits_idx, shape=(self.batch_size,))
        correct_preds = tf.equal(tf.to_int32(tf.argmax(labels, axis=1)), logits_idx)
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) / self.batch_size

        return accuracy    