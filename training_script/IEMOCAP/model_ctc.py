#!/usr/bin/env python
# encoding: utf-8

import os
import time
import data
import config
import os.path
import numpy as np
import tensorflow as tf
np.set_printoptions(threshold=np.inf)

classes={0:'ang',1:'hap',2:'neu',3:'sad'}

class RecognitionNN(object):
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope('weights'):
                self.weights = {
                    'W_conv1': tf.get_variable('W_conv1', [10,1,1,4],
                                                initializer=tf.truncated_normal_initializer(stddev=0.1)),
                    'W_conv2': tf.get_variable('W_conv2', [5,1,4,8],
                                                initializer=tf.truncated_normal_initializer(stddev=0.1)),
                    'W_conv3': tf.get_variable('W_conv3', [3,1,8,16],
                                                initializer=tf.truncated_normal_initializer(stddev=0.1))
                }
            
            with tf.variable_scope('biases'):
                self.biases = {
                    'b_conv1':tf.get_variable('b_conv1', [4],
                                                initializer=tf.constant_initializer(0, dtype=tf.float32)),
                    'b_conv2':tf.get_variable('b_conv2', [8],
                                                initializer=tf.constant_initializer(0, dtype=tf.float32)),
                    'b_conv3':tf.get_variable('b_conv3', [16],
                                                initializer=tf.constant_initializer(0, dtype=tf.float32))
                }
                
            # input_x.shape: [batch_size, max_step, fea_dim]
            self.input_x = tf.placeholder(tf.float32,shape=[None,None,config.fea_dim],name="inputs_x")
            # input_y.shape:[batch_size, emo_num]
            self.input_y = tf.placeholder(tf.int32,shape=[None,None],name="labels_y")
            # seq_len: [batch_size]
            self.seq_len = tf.placeholder(tf.int32, shape=[None], name="feature_len") # nums of frames
            # label_len : [batch_szie]
            self.lab_len = tf.placeholder(tf.int32, shape=[None], name="label_len")  # [A A A ]
            # batch_szie 
            self.batch_size = tf.placeholder(tf.int32,[], name="batch_size")
            # training or testing label
            self.is_train = tf.placeholder(tf.bool, None)
            self.keep_prob=tf.placeholder(tf.float32,name="keep_prob")
            
            self.mu=tf.placeholder(tf.float32,shape=[config.fea_dim],name="mu")
            self.var=tf.placeholder(tf.float32,shape=[config.fea_dim],name="var")
            
            fea_norm=tf.nn.batch_normalization(self.input_x, self.mu, self.var, 0, 2, 0.001, name="normalize")
            self.input_x_bn = fea_norm 
            
            with tf.name_scope('cnn_net'):
                # x_data.shape:[batch_size, max_step, fea_dim, 1]
                self.x_data = tf.reshape(self.input_x_bn, [self.batch_size, -1, config.fea_dim, 1])
                # first convolution and pooling
                with tf.name_scope('conv1'):
                    print('self.x_data:', self.x_data)
                    conv1 = tf.nn.conv2d(self.x_data, self.weights['W_conv1'], strides=[1,1,1,1], padding='SAME')   
                    h_conv1 = tf.nn.relu(tf.nn.bias_add(conv1, self.biases['b_conv1']))
                    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,3,1,1], strides=[1,2,1,1], padding='SAME')
                    print("h_pool1:", h_pool1)
                # second convolution and pooling
                with tf.name_scope('conv2'):
                    conv2 = tf.nn.conv2d(h_pool1, self.weights['W_conv2'], strides=[1,1,1,1], padding='SAME')
                    h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2, self.biases['b_conv2']))
                    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,3,1,1], strides=[1,2,1,1], padding='SAME')
                    print("h_pool2:", h_pool2)
                # third convolution and pooling
                with tf.name_scope('conv3'):
                    conv3 = tf.nn.conv2d(h_pool2, self.weights['W_conv3'], strides=[1,1,1,1], padding='SAME')
                    h_conv3 = tf.nn.relu(tf.nn.bias_add(conv3, self.biases['b_conv3']))
                    h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1,3,1,1], strides=[1,2,1,1], padding='SAME')
                    print("h_pool3:", h_pool3)
                self.cnn_result = h_pool3
                print("self.cnn_result:", self.cnn_result) # [batch_size, frames_nums, fea_dim, 16]
            
            shape=self.cnn_result.get_shape().as_list()
            print('shape:', shape)
            self.cnn_results = tf.reshape(self.cnn_result, [self.batch_size, -1, shape[2]*16]) 
            print("self.cnn_results:", self.cnn_results)
            
            self.new_seq_len = tf.ceil((tf.to_float(self.seq_len))/8)
            self.new_seq_len = tf.cast(self.new_seq_len, tf.int32)
            
            with tf.name_scope('lstm_net'):
                count = -1
                hidden_layer = []
                with tf.name_scope('lstm_layer'):
                    for unit_num in config.lstm_hidden_size:
                        count = count+1
                        with tf.name_scope('lstm_cell_'+str(count)):
                            lstm_cell = tf.contrib.rnn.LSTMCell(unit_num)
                        hidden_layer.append(lstm_cell)
                
                stack = tf.contrib.rnn.MultiRNNCell(hidden_layer, state_is_tuple=True)
                init_state = stack.zero_state(self.batch_size, dtype=tf.float32)
                outputs, last_states = tf.nn.dynamic_rnn(stack, self.cnn_results, self.new_seq_len, initial_state=init_state, dtype=tf.float32, time_major=False) # tf.ceil(tf.to_float(self.seq_len))
                print('outputs:', outputs) # [batch_size, frame_nums, 256]
                print('last_states:', last_states)
                #h_output = last_states[-1][-1]
                h_output = tf.reshape(outputs, [-1,config.lstm_hidden_size[-1]]) #  [batch_size*frame_nums, 256]
                self.h_output = h_output
                print("self.h_output:", self.h_output)
                
            with tf.name_scope('dense_net'):
                # full_conn  f_dense.shape:[batch_size, config.full_connect_layer_unit]
                f_dense = tf.contrib.layers.fully_connected(self.h_output, config.full_connect_layer_unit, activation_fn=None, scope='full_conn')
                if config.do_batchnorm:
                    self.f_dense = tf.contrib.layers.batch_norm(f_dense, decay=0.99, center=True, scale=True, updates_collections=None, is_training=self.is_train, scope='bn')
                else:
                    self.f_dense = f_dense
                #logits.shape: [batch_size, config.emo_num]
                logits = tf.contrib.layers.fully_connected(self.f_dense, config.class_num, activation_fn=None, scope='logits')  
                #[batch_size, timestep, config.emo_num]
                self.logit = tf.reshape(logits, [self.batch_size, -1, config.class_num])# [batch_size*frame_nums, 4]--> [batch_size, frame_nums, 256]
                logits = tf.transpose(self.logit, (1, 0, 2))
                self.logits = logits
                #[timesteps ,batch_size, 5]
                print("logits:", self.logits)
                
            with tf.name_scope('accuracy'):
                self.global_step = tf.Variable(0, trainable=False)
                targets = tf.contrib.keras.backend.ctc_label_dense_to_sparse(self.input_y, self.lab_len)  #framenums*0.03
                loss = tf.nn.ctc_loss(labels=targets, inputs=self.logits, sequence_length=self.new_seq_len) # framenums/8
                self.cost = tf.reduce_mean(loss)
                self.optimizer = tf.train.AdamOptimizer(config.initial_learning_rate).minimize(self.cost, self.global_step)
                self.decoded, log_prob = tf.nn.ctc_beam_search_decoder(self.logits, self.new_seq_len, merge_repeated=False)
                self.decoded_dense=tf.sparse_tensor_to_dense(self.decoded[0],default_value=(config.class_num-1))
                dis = tf.edit_distance(tf.cast(self.decoded[0], tf.int32), targets)
                self.acc = tf.reduce_mean(dis)
                
            if config.out_model:
                saver = tf.train.Saver(max_to_keep=30)
                self.saver = saver
                
    def m_print(self, str, log_file):
        print(str) 
        if not os.path.exists(log_file):
            os.mknod(log_file)
        if config.out_log:
            with open(log_file,'a') as fout:
                fout.write(str+"\n")
            
    def calculate_ua(self,labels, logits):
        total_array = np.zeros(config.emo_num)
        right_array = np.zeros(config.emo_num)
        
        maxindex_labels = labels
        maxindex_logits = logits 
        for index in range(len(maxindex_labels)):  	# len(maxindex_labels) = batch_size
            total_array[maxindex_labels[index]] += 1
            if maxindex_logits[index] == maxindex_labels[index]:
                right_array[maxindex_labels[index]] += 1
        acc_ua = right_array/total_array
                
        print('right_array:', right_array)
        print('total_array:', total_array)
        
        acc_ua[np.isnan(acc_ua)] = 0
        print('acc_ua:',acc_ua)
        return np.mean(acc_ua)     # right_array/total_array: 是一个[] 列表， 取其平均值作为ua
            
    def get_pl(self, hepothesis_seq, log_file):
        rt=[]
        if hepothesis_seq==[]:
            return rt
        entry_num=np.shape(hepothesis_seq)[0]
        for i in range(entry_num):
            h=hepothesis_seq[i]
            if len(h)==0:
                self.m_print("get_pl():h==0",log_file)
                continue
            c=np.bincount(h)
            if config.label_add_blank:
                d=c[:(config.emo_num-1)]
            else:
                d=c[:config.emo_num]
            p_l=np.argmax(d)
            rt.append(p_l)
        return rt
            
    def get_batch_acc(self, decoded_dense, true_label_list, log_file, flag, test_time, vali_num):
        if decoded_dense == []:
            return 0, 0, 0
        right_rate = 0
        right_num = 0
        total_num = 0
        total_num = np.shape(decoded_dense)[0]
        
        total_array = np.zeros(config.emo_num)
        right_array = np.zeros(config.emo_num)
        
        true = []
        pre = []
        
        for i in range(total_num):
            wav_frame_logits = decoded_dense[i]
            if len(wav_frame_logits) == 0:
                self.m_print('get_batch_acc: wav_frame_logits==0', log_file)
                exit()
                continue
            if config.label_add_blank:
                label = true_label_list[i][1]
            else:
                label = true_label_list[i][0]
            labelbin = np.bincount(wav_frame_logits)
            if config.label_add_blank:
                label_list = labelbin[:(config.emo_num-1)]
            else:
                label_list = labelbin[:config.emo_num]
            temp = sum(label_list)
            label_list = label_list /float(temp)
            p_l = np.argmax(label_list)
            
            total_array[p_l] += 1
            if p_l == label:
                right_num = right_num+1
                right_array[p_l] += 1
            
            true.append(label)
            pre.append(p_l)

        if flag == 'test':
            test_end_time = time.time()-test_time
            self.m_print("vali_num:%d"%vali_num, log_file)
            self.m_print("test_time:%fs"%test_end_time, log_file)
            
        if total_num != 0:
            right_rate = float(right_num)/total_num
        
        acc_ua = 0
        if flag == 'test':
            acc_ua = right_array / total_array
            acc_ua[np.isnan(acc_ua)] = 0
            print('right_array:', right_array)
            print('total_array:', total_array)
            print('acc_ua:',acc_ua)
        
        self.m_print('true_label:%s'%true, log_file)
        self.m_print('predict_label:%s'%pre, log_file)
        return right_rate, np.mean(acc_ua), right_num, total_num
            
    def test_model(self, sess, val_set, mu_, var_, log_file, curr_epoch):
        r_num=0
        t_num=0
        r_rate=0
        p_l=[]
        t_l=[]
        p_key=[]
        p_seq=[]
        
        decoded_dense = []
        true_label = []
        
        vali_num = 0
        test_time = time.time()
        
        while True:
            keys1, val_features, seq_len1, val_labels, lab_len1, batch_num = val_set.next_batch(config.batch_size)
            if batch_num==0:
                break
            print("vali batch_num:", batch_num)
            
            vali_num += batch_num
            val_feed = {self.input_x:val_features,
                        self.seq_len:seq_len1,
                        self.input_y:val_labels,
                        self.lab_len:lab_len1,
                        self.batch_size:batch_num,
                        self.is_train:False,
                        self.mu:mu_,
                        self.var:var_}
            
            b_decoded = sess.run([self.decoded_dense], feed_dict=val_feed)
            for i in range(np.shape(b_decoded)[1]):
                decoded_dense.append(b_decoded[0][i])
                true_label.append(val_labels[i])
            
        val_wa, val_ua, val_right_num, val_total_num = self.get_batch_acc(decoded_dense, true_label, log_file, 'test', test_time, vali_num)
        return val_wa ,val_ua
        
    def train(self, train_data_path, vali_data_path, log_file=None, model_dir=None):
        cv_time_start = time.time()
        train_set = data.CDataSet(train_data_path, "train", shuffle=True)
        val_set = data.CDataSet(vali_data_path, "vali", shuffle=False)
        
        self.m_print("###train num:%d"%train_set.sample_num,log_file)
        self.m_print("###val num:%d"%val_set.sample_num,log_file)
        val_max_ua = -1
        
        configs = tf.ConfigProto()
        configs.gpu_options.allow_growth=True
        with tf.Session(config=configs, graph=self.graph) as sess:
            '''
            if config.do_visualization:
                tf.summary.scalar('Loss', self.cost)
                tf.summary.scalar('acc',self.acc)
                merged_summary_op = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter('/data/mm0105.chen/wjhan/dzy/LSTM+CTC/CCLDNN/photo/', tf.get_default_graph())
            '''    
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            
            val_acc_cv_wa = []
            val_acc_cv_ua = []
            train_acc_cv_wa = []
            train_acc_cv_ua = []
            
            for curr_epoch in range(config.num_epoches):
                self.m_print("###Epoch %d begin"%curr_epoch, log_file)
                self.m_print("---", log_file)
                
                epoch_time_start = time.time()
                
                i = 0
                epoch_loss = 0
                epoch_right_num = 0
                epoch_total_num = 0
                # train 
                while True:
                    batch_start_time = time.time()
                    keys1,features1,seq_len1,labels,lab_len1,batch_num=train_set.next_batch(config.batch_size)
                    
                    if batch_num == 0:
                        break
                    
                    if batch_num < config.batch_size:
                        print("train batch_num:", batch_num)
                        
                    train_feed = {self.input_x: features1, 
                                    self.seq_len:seq_len1,
                                    self.input_y: labels,
                                    self.lab_len: lab_len1,
                                    self.batch_size:batch_num,
                                    self.is_train:True,
                                    self.mu:train_set.mu,
                                    self.var:train_set.var}
                    
                    if config.do_visualization:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        _, batch_cost, batch_acc, step, summary= sess.run([self.optimizer, self.cost, self.acc, self.global_step, merged_summary_op], 
                                                                            feed_dict=train_feed, options=run_options,run_metadata=run_metadata)
                    
                        train_writer.add_run_metadata(run_metadata, 'step%05d' % step)
                        train_writer.add_summary(summary,step)
                    else:
                        
                        _, batch_cost, batch_acc, step, batch_decoded_dense= sess.run([self.optimizer, self.cost, self.acc, self.global_step, self.decoded_dense],feed_dict=train_feed)
                        
                        
                    batch_rate, batch_rate_ua, batch_right_num, batch_total_num = self.get_batch_acc(batch_decoded_dense, labels, log_file, 'train',0,0)
                    
                    epoch_loss = (epoch_loss*epoch_total_num + batch_cost*batch_num ) / (epoch_total_num + batch_num)
                    epoch_right_num = epoch_right_num+batch_right_num
                    epoch_total_num = epoch_total_num+batch_num
                    
                    self.m_print("###epoch %d"%curr_epoch, log_file)
                    self.m_print("###global_step:%d"%step, log_file)
                    self.m_print("###batch_cost:%f"%batch_cost, log_file)
                    self.m_print("###batch_acc:%f"%batch_acc, log_file)
                    self.m_print("###batch_rate:%f"%batch_rate, log_file)
                    self.m_print("###batch_num:%d"%batch_num, log_file)
                    self.m_print("###batch_right_num:%d"%batch_right_num, log_file)
                    self.m_print("###batch_time:%fs"%(time.time()-batch_start_time), log_file)
                    
                self.m_print("****************************", log_file)
                
                # test 
                val_wa, val_ua = self.test_model(sess, val_set, train_set.mu, train_set.var, log_file, curr_epoch)
                if val_ua>val_max_ua:
                    val_max_ua = val_ua
                    if config.out_model:
                        model_file = os.path.join(model_dir, 'model.ckpt')
                        rt = self.saver.save(sess, model_file.replace('.ckpt', '_'+str(curr_epoch)+'.ckpt'))
                        self.m_print("model saved in %s"% rt, log_file)
                
                self.m_print('epoch_recognition_correct_nums:%d'%epoch_right_num, log_file)
                self.m_print('epoch_total_nums:%d'%epoch_total_num, log_file)
                train_acc_wa = (float(epoch_right_num) / float(epoch_total_num))
                
                # wa: epoch_correct_num / epoch_total_num
                val_acc_cv_wa.append(val_wa)
                val_acc_cv_ua.append(val_ua)
                train_acc_cv_wa.append(train_acc_wa)
                
                self.m_print('\nEpoch %d finished'%curr_epoch, log_file)
                self.m_print('Epoch loss %f'%epoch_loss, log_file)
                self.m_print('Epoch train_acc_wa %f'%(train_acc_wa), log_file)
                self.m_print('Epoch %d val_acc_wa %f'%(curr_epoch, val_wa), log_file)
                self.m_print('Epoch %d val_acc_ua %f'%(curr_epoch, val_ua), log_file)
                self.m_print('Epoch_time_cost:%fs'%(time.time()-epoch_time_start), log_file)
                
                
                
            self.m_print('-------------------------------------------------------------------------', log_file)
            self.m_print("###cv finished", log_file)
            self.m_print('--------UA------------', log_file)
            self.m_print('val_acc_cv_ua_max:%f in epoch %d'%(val_acc_cv_ua[np.argmax(val_acc_cv_ua)], np.argmax(val_acc_cv_ua)), log_file)
            self.m_print('val_acc_cv_wa_max:%f'%(val_acc_cv_ua[np.argmax(val_acc_cv_wa)]), log_file)
            
            self.m_print('--------WA------------',log_file)
            self.m_print('val_acc_cv_wa_max:%f in epoch %d'%(val_acc_cv_wa[np.argmax(val_acc_cv_wa)], np.argmax(val_acc_cv_wa)), log_file)
            self.m_print('train_acc_cv_wa_max:%f in epoch %d'%(train_acc_cv_wa[np.argmax(val_acc_cv_wa)], np.argmax(val_acc_cv_wa)), log_file)
            self.m_print('val_acc_cv_ua_max:%f'%(val_acc_cv_ua[np.argmax(val_acc_cv_wa)]), log_file)
            
            self.m_print('cv_time_cost:%fs'%(time.time()-cv_time_start), log_file)
            if config.do_visualization:
                train_writer.close()
                
            
    
    
    
    
    
    
    
    
    
    
    
    
    
