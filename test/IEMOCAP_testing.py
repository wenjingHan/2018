#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import numpy as np
import tensorflow as tf

emo_label={0:'ang',1:'hap',2:'neu',3:'sad'}


def read_fea(fea_path):
    fea = []
    with open(fea_path, 'r') as fin:
        data = fin.readlines()[0:]
    for temp in data:
        temp = temp.strip().split(";")[2:]
        fea.append(temp)
    return np.array(fea,dtype=np.float), len(data)


def read_mu_var(script_path):
    mu = []; var = []
    with open(os.path.join(script_path,"average_cv0.txt"), 'r') as fout:
        for line in fout.readlines():
            for value in line.strip().split(",")[0:]: mu.append(float(value))
    with open(os.path.join(script_path,"variance_cv0.txt"), 'r') as fout:
        for line in fout.readlines():
            for value in line.strip().split(",")[0:]: var.append(float(value))
    return mu, var


def run_model(sess, fea_path, script_path):
    print("\nrun predict model......")
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('inputs_x:0')
    y = graph.get_tensor_by_name('accuracy/SparseToDense:0')
    x_lengths = graph.get_tensor_by_name('feature_len:0')
    batch_size = graph.get_tensor_by_name('batch_size:0')
    is_train = graph.get_tensor_by_name('Placeholder:0')
    mu = graph.get_tensor_by_name('mu:0')
    var = graph.get_tensor_by_name('var:0')
    
    
    trial_data, fea_len = read_fea(fea_path)
    train_set_mu, train_set_var = read_mu_var(script_path)
    predicts = sess.run(y, feed_dict={x:[trial_data], 
                                        x_lengths:[fea_len],
                                        batch_size:1,
                                        mu:train_set_mu,
                                        var:train_set_var,
                                        is_train:False})
    
    pre_labelbin = np.bincount(predicts[0])
    label_list = pre_labelbin[:4]
    predict_label = np.argmax(label_list)
    print("Predict result: %s"%emo_label[predict_label])
    sess.close()


def start_session_ckpt(model_dir):
    print("\nstart_session_ckpt(%s)"%model_dir)
    print("[*] Reading checkpoints...")
    
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print("Loading success...")
        print(ckpt_name)
    else:
        print("[*] Failed to find a checkpoint, exit")
        exit()
    
    print("tensorflow version: %s"%str(tf.__version__))
    print(os.path.join(model_dir, ckpt_name)+'.meta')
    saver = tf.train.import_meta_graph(os.path.join(model_dir,ckpt_name) + '.meta',clear_devices=True)
    sess = tf.InteractiveSession()
    init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    sess.run(init)
    saver.restore(sess, os.path.join(model_dir, ckpt_name))
    return sess


if __name__ == "__main__":
    # python IEMOCAP_testing.py model_dir wav_name
    os.environ['CUDA_VISIBLE_DEVICES'] = '14'
    model_dir = sys.argv[1]
    fea_path = sys.argv[2]
    script_path = sys.path[0]

    sess = start_session_ckpt(model_dir)
    run_model(sess, fea_path, script_path)
