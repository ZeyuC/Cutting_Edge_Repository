#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 16:05:19 2020

@author: chenzeyu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import random
import tensorflow_probability as tfp

from sklearn.model_selection import train_test_split

#Reset all the graph 
tf.reset_default_graph()

## Import our raw data
data_close = pd.read_csv('data.csv')

n_time = data_close.shape[0]
n_stocks = data_close.shape[1]


# Data treatment ,fill the NA and calculate the daily reture
for j in range(0,n_stocks):
        data_close.iloc[:,j]=data_close.iloc[:,j].fillna(data_close.iloc[:,j].mean())
#rendements = np.log(data_close.values[1:,:]/data_close.values[:-1,:])
#rendements = np.float32(rendements)
X=(data_close)/np.std(data_close)
#Split the raw data to two part train and test

X_data =  X.iloc[:,0:10] 
#X_train, X_test = train_test_split(X_data, test_size = 0.35,shuffle=False)
X_train, X_test = train_test_split(X_data, test_size = 0.35,random_state=42)
#Constants declaration 
Y_size = X_train.shape[0]     #the number of date we will used for one network training
X_size = X_train.shape[1]     #X_size is the number of stock
epochs = 6000             #the number of iteration

##############################################
##############################################

## Generate noise
def sample_noise_uniform(n=Y_size, dim=X_size):        
    return np.random.uniform(-1,1,(n,dim))

def sample_noise_Gaus(n=Y_size, dim=X_size):        
    return np.random.normal(0,1,(n,dim))

 
def generator(Z,nb_neurone=64,reuse=False):
    """ generator structure
    Args:
        Z: The noise that is uniform or Gaussain
        nb_neurone : number of neurone of one layer
        reuse: False means create a new variable,True means reuse the existing one 
    """
    with tf.variable_scope("GAN/Generator",reuse=reuse):
        h1 = tf.layers.dense(Z,nb_neurone,activation=tf.nn.leaky_relu)
        #h2 = tf.layers.dense(h1,nb_neurone,activation=tf.nn.leaky_relu)
        output = tf.layers.dense(h1,X_size)
    return output


def discriminator(X,nb_neurone=[64,16],reuse=False):
    """ generator structure
    Args:
        X: The real data or generated data 
        nb_neurone : number of neurone of one layer
        reuse: False means create a new variable,True means reuse the existing one 
    """
    with tf.variable_scope("GAN/Discriminator",reuse=reuse):
        h1 = tf.layers.dense(X,nb_neurone[0],activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,nb_neurone[1],activation=tf.nn.leaky_relu)
        h3 = tf.layers.dense(h2,2)
        output = tf.layers.dense(h3,1)

    return output

X = tf.placeholder(tf.float32,[None,X_size])
Z = tf.placeholder(tf.float32,[None,X_size])



gen_sample = generator(Z)
real_logits = discriminator(X)
gen_logits = discriminator(gen_sample,reuse=True)

#corr = tf.transpose(tfp.stats.correlation(gen_sample))
#corr_loss = tf.reduce_sum(corr)- tf.reduce_sum(tf.diag_part(corr))

#dis_loss = min E[-log(D(X))] + E[log(1-D(G(Z)))] := real_loss + gen_loss
#sigmoid_cross_entropy_with_logits(x,z) = z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
r_loss=tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits,labels=tf.ones_like(real_logits))
g_loss=tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_logits,labels=tf.zeros_like(gen_logits))
disc_loss = tf.reduce_mean(r_loss + g_loss)


#gen_loss = min E[log(1-D(G(Z)))] =  max  E[log D(G(Z)] = min - E[log(D(G(Z)))]

gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_logits,labels=tf.ones_like(gen_logits)))


#Define the Optimizer with learning rate  0.001
gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")
gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001,decay=0.95, momentum=0.0, epsilon=1e-10).minimize(gen_loss,var_list = gen_vars) 
disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001,decay=0.95, momentum=0.0, epsilon=1e-10).minimize(disc_loss,var_list = disc_vars) 
#gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss,var_list = gen_vars) 
#disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss,var_list = disc_vars) 



sess = tf.Session()
sess.run(tf.global_variables_initializer())
nd_steps=4 #entrainer plus de dis que gen
ng_steps=8
#Training process

X_batch = X_train
with tf.device('/device:GPU:0'):
    for i in range(epochs):
        Z_batch = sample_noise_Gaus(Y_size, X_size)
        #ind_X = random.sample(range(Y_size),Y_size)
        for _ in range(nd_steps):
            _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, Z: Z_batch})
    #rrep_dstep, grep_dstep = sess.run([r_rep, g_rep], feed_dict={X: X_batch, Z: Z_batch})

        for _ in range(ng_steps):
            _, gloss = sess.run([gen_step, gen_loss], feed_dict={Z: Z_batch})

        if i%100==0:
            print ("Iterations:" ,i,"Discriminator loss: ",dloss,"Generator loss:" , gloss)
        

        
#Generate data with our generator by feeding Z 
pred=sess.run(gen_sample,feed_dict={Z: Z_batch})

#Check if generator cheated discriminator by checking if Prob_real and
#Prob_pred are closed to 0.5
y_real=sess.run(real_logits,feed_dict={X: X_batch})

Prob_real=sess.run(tf.sigmoid(y_real))

y_pred=sess.run(real_logits,feed_dict={X: pred})

Prob_pred=sess.run(tf.sigmoid(y_pred))


#Check the cov matrix (problem need to solve)
np.set_printoptions(suppress=True)

Mean_pred = np.mean(np.transpose(pred),axis=1)
Mean_X = np.mean(np.transpose(X_batch),axis=1)
Cov_pred = np.around(np.cov(np.transpose(pred)), decimals=3)
#print(np.around(np.cov(np.transpose(pred)), decimals=2))
Cov_X = np.around(np.cov(np.transpose(X_batch)), decimals=3)
#print(np.around(np.cov(np.transpose(X_batch)), decimals=2))

Corr_pred = np.around(np.corrcoef(np.transpose(pred)), decimals=3)
Corr_X = np.around(np.corrcoef(np.transpose(X_batch)), decimals=3)