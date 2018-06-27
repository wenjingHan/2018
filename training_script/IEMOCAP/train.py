#!/usr/bin/env python
# encoding: utf-8

import os
import time
import config
import datetime
import model_ctc

def train_ctc(train_data_path, val_data_path, cv_id):
    print("begin training........")
    RecognitionNN = model_ctc.RecognitionNN()
    
    if config.out_log:
        timebase = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        log_dir = os.path.join(config.log_dir, timebase)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file = os.path.join(log_dir, timebase+'.log')
        RecognitionNN.m_print('\n\nlog saved in %s'%log_file, log_file)
    
    if config.out_model:
        model_dir = os.path.join(config.model_dir, timebase)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        RecognitionNN.m_print('model saved in %s'%model_dir, log_file)
    else:
        model_dir = ""
    
    RecognitionNN.m_print('>######trainning CV %d'%cv_id, log_file)
    RecognitionNN.m_print('>######Net Structure######', log_file)
    RecognitionNN.m_print('>###Type: cnn+lstm+dense+bn+dense+ ctc', log_file)
    RecognitionNN.m_print('>###Input dim: %d'%config.fea_dim, log_file)
    RecognitionNN.m_print('>###OutPut dim: %d'%config.class_num, log_file)
    RecognitionNN.m_print('>###LSTM Hidden size: %s'%str(config.lstm_hidden_size), log_file)
    RecognitionNN.m_print('>###Full-conn Hidden size: %s'%str(config.full_connect_layer_unit), log_file)
    
    RecognitionNN.m_print('>###emo_num: %d'%config.emo_num, log_file)
    RecognitionNN.m_print('>###batch_size: %d'%config.batch_size, log_file)
    RecognitionNN.m_print('>###num_epoches: %d'%config.num_epoches, log_file)
    
    RecognitionNN.m_print('>###Shuffle: %s'%config.shuffle, log_file)
    RecognitionNN.m_print('>###lr_decay: %s'%config.lr_decay, log_file)
    RecognitionNN.m_print('>###do_dropout: %s'%config.do_dropout, log_file)
    RecognitionNN.m_print('>###do_batchnorm: %s'%config.do_batchnorm, log_file)
    RecognitionNN.m_print('>###do_visualization: %s'%config.do_visualization, log_file)
    RecognitionNN.m_print('>###initial_learning_rate: %f'%config.initial_learning_rate, log_file)
    RecognitionNN.m_print('>###Train set: \n%s'%str(train_data_path), log_file)
    RecognitionNN.m_print('>###vali set: \n%s'%str(val_data_path), log_file)
    
    
    RecognitionNN.train(train_data_path, val_data_path, log_file, model_dir)
    
if __name__ == '__main__':
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '10'
    
    for i in range(2):
        i = i + 8
        val_data_path = []
        train_data_path = []
        
        for cv_id in range(10):
            if cv_id == i:
                
                val_data_path.append(os.path.join("../../features/238_IEMOCAP/CV0-10/", str(cv_id)+'.txt'))
                continue
            train_data_path.append(os.path.join("../../features/238_IEMOCAP/CV0-10/", str(cv_id)+'.txt'))
        train_ctc(train_data_path, val_data_path, i)
        
