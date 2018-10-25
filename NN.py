import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix

TF_SEED = 777

class NN:
    def __init__(self, session:tf.Session, n_feature, n_lookback,dropout_keep_prob,lr,lr_decay, name):
        self.sess = session
        self.n_lookback = n_lookback
        self.n_feature = n_feature
        self.n_class = 9
        self.dropout_keep_prob = dropout_keep_prob
        self.name = name
        
        with tf.name_scope(name):
            self.x = tf.placeholder(tf.float32, [None, n_lookback, n_feature],name='x')
            self.y = tf.placeholder(tf.float32, [None, self.n_class],name='y_onehot')
            self.is_training = tf.placeholder(tf.bool)
            self.tf_dropout_keep_prob = tf.placeholder(tf.float32)
            global_step = tf.Variable(0, trainable=False)
            
            decaying_lr = tf.train.exponential_decay(lr, global_step, 10000, lr_decay)
            x = self.x
            output = self.hidden_layers(x)
            
            self.y_softmax = tf.nn.softmax(output)
            self.y_pred = tf.expand_dims(tf.argmax(output, axis = -1), 1)
            
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=output))
#             self.optimizer = tf.contrib.opt.AdamOptimizer(lr)
            self.optimizer  = tf.train.AdamOptimizer(decaying_lr)
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.optimizer.minimize(self.loss, global_step=global_step)
    def hidden_layers(self,x):
        """
         x: shape=[-1, n_lookback, n_feature]
         return: logit tensor (shape=[-1, n_class])
        """
        raise NotImplementedError("model construct")
        pass
    def run_batch(self, x, y, batch_size, is_training):
        #for inference
        if batch_size == -1:
            batch_size = len(x)
        
        total_loss = 0
        y_preds = []
        
#         print(is_training)
        total_steps = len(x) // batch_size
        last_batch_size = len(x) - total_steps*batch_size
        if last_batch_size != 0:
#             print("last batch size is ", last_batch_size)
            total_steps +=1
        if is_training:
            for i in range(total_steps):
                x_batch = x[i * batch_size:(i+1) * batch_size]
                y_batch = y[i * batch_size:(i+1) * batch_size]
                _, loss, y_pred = self.sess.run([self.train_op, self.loss, self.y_pred],
                             feed_dict={
                                 self.x: x_batch,
                                 self.y: y_batch,
                                 self.tf_dropout_keep_prob:self.dropout_keep_prob,
                                 self.is_training:True})
                total_loss += loss
#                 print(y_pred, y_batch)
                y_preds.append(y_pred)
        else:
            for i in range(total_steps):
                x_batch = x[i * batch_size:(i+1) * batch_size]
                y_batch = y[i * batch_size:(i+1) * batch_size]
                loss, y_pred = self.sess.run([self.loss, self.y_pred],
                             feed_dict={
                                 self.x: x_batch,
                                 self.y: y_batch,
                                 self.tf_dropout_keep_prob:1.,
                                 self.is_training:False})
                total_loss += loss
                y_preds.append(y_pred)
        y_preds = np.concatenate(y_preds)
        
        return total_loss/total_steps, y_preds
    
    def predict_problem(self, x_list):
        y_pred = self.sess.run([self.y_pred],
                                             feed_dict = {
                                                 self.x  :x_list,
                                                 self.tf_dropout_keep_prob:1.,
                                                 self.is_training: False                                            
                                             })
        return y_pred
        
    def train(self, train_data, BATCH_SIZE, EPOCH):
        for i in range(EPOCH):
            print("Name : {} , EPOCH : {}".format(self.name, i))
            train_loss, train_pred = self.run_batch(train_data['train']['x'],train_data['train']['y']
                                           ,BATCH_SIZE, is_training = True)

            val_loss, val_pred = self.run_batch(train_data['val']['x'],train_data['val']['y']
                                           ,BATCH_SIZE, is_training = False)
            y_train_real=  np.reshape(np.argmax(train_data["train"]["y"],axis = 1),(-1,1))
            train_acc = np.sum(np.equal(train_pred, y_train_real)) / len(y_train_real)
            
#             print(train_pred)
            y_val_real=  np.reshape(np.argmax(train_data["val"]["y"],axis = 1),(-1,1))
            val_acc = np.sum(np.equal(val_pred, y_val_real)) / len(y_val_real)
            print("loss -> train : {} , validation : {}".format(train_loss, val_loss))
            print("acc -> train : {} , validation : {}".format(train_acc, val_acc))
            
        # predict problem data (test data set) at final epoch
        for i in range(0,9):
            pred = self.predict_problem(train_data["test"][i])
            pred = np.ravel(pred)
            predict_driver = []
            for j in range(0,9):
                predict_driver.append(np.count_nonzero(pred==j))
            print(i," : ", predict_driver)
            
            
class Dense(NN):
    def __init__(self, session:tf.Session, n_feature, n_lookback,dropout_keep_prob,lr,lr_decay, name):
        super().__init__(session, n_feature, n_lookback, dropout_keep_prob, lr,lr_decay, name)
    
    def hidden_layers(self, x):
        print(x)
        X = tf.layers.flatten(x)
        layer1 = tf.contrib.layers.fully_connected(X, 1024, activation_fn=tf.nn.relu,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(TF_SEED))
        layer1 = tf.layers.dropout(layer1, rate=1 - self.dropout_keep_prob, training=self.is_training, seed=TF_SEED)
                

        layer2 = tf.contrib.layers.fully_connected(layer1, 1024, activation_fn=tf.nn.relu,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(TF_SEED))
        layer2 = tf.layers.dropout(layer2, rate=1 - self.dropout_keep_prob, training=self.is_training, seed=TF_SEED)
        
        layer3 = tf.contrib.layers.fully_connected(layer2, 1024, activation_fn=tf.nn.relu,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(TF_SEED))
        layer3 = tf.layers.dropout(layer3, rate=1 - self.dropout_keep_prob, training=self.is_training, seed=TF_SEED)
        
        layer4 = tf.contrib.layers.fully_connected(layer3, 1024, activation_fn=tf.nn.relu,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(TF_SEED))
        layer4 = tf.layers.dropout(layer4, rate=1 - self.dropout_keep_prob, training=self.is_training, seed=TF_SEED)
        
        layer5 = tf.contrib.layers.fully_connected(layer4, 1024, activation_fn=tf.nn.relu,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(TF_SEED))
        layer5 = tf.layers.dropout(layer5, rate=1 - self.dropout_keep_prob, training=self.is_training, seed=TF_SEED)
                
                
        output = tf.contrib.layers.fully_connected(layer5, 9, activation_fn=None,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(TF_SEED))
        
        
        return output
    
    
class CNN(NN):
    def __init__(self, session:tf.Session, n_feature, n_lookback,dropout_keep_prob,lr,lr_decay, name):
        super().__init__(session, n_feature, n_lookback, dropout_keep_prob, lr,lr_decay, name)
    
    def hidden_layers(self, x):
        print(x)
        X = tf.reshape(x, [-1, self.n_lookback, self.n_feature, 1])
        
        conv1 = tf.layers.conv2d(X, filters=64, kernel_size= [1, 3], 
                                kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(TF_SEED),
                                activation=tf.nn.relu)
        conv1 = tf.layers.dropout(conv1, rate= 1-self.dropout_keep_prob, training=self.is_training)
        
        conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=[self.n_lookback,1],
                                 kernel_initializer= tf.contrib.layers.xavier_initializer_conv2d(TF_SEED),
                                 activation=tf.nn.relu)
        conv2 = tf.layers.dropout(conv2, rate= 1-self.dropout_keep_prob, training=self.is_training)
            
            
        flattenl  = tf.contrib.layers.flatten(conv2)
        output = tf.contrib.layers.fully_connected(flattenl, 9, activation_fn=None,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(TF_SEED))
        
        
        return output