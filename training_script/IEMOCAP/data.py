#!/usr/bin/env python
# encoding: utf-8

import time
import os
import config
import random
import numpy as np
import matplotlib.image as mpimg
# natual: 2   anger:0  sad: 3    hap：1

emo_class = {'ang':0, 'hap':1, 'neu':2, 'sad':3, 'exc':1}

class CDataSet:
    def __init__(self, filelist_path, flag, shuffle=True):
        self.data={}
        self.sample_names=[]
        self.rest_samples = []
        self.batch_id = 0
        self.is_shuffle = shuffle
        self.sample_num=0
        self.frame_num=0
        self.mu=0
        self.var=0
        
        self.fea_spec = []
        self.mu_spec = 0
        self.var_spec = 0
        if flag == "spec":
            for flist in filelist_path:
                self.add_specgram(flist)
        else:
            for flist in filelist_path:
                self.add_data(flist)
        self.sample_heap=self.sample_names

    def add_data(self,filelist):
        with open(filelist,'r') as fin:
            files=fin.readlines()
        for file in files:
            file=file.strip()
            pack=[]
            fea=[]
            with open(file,'r') as fin:
                data=fin.readlines()[0:]
            fea_len=len(data)   # nums of frames
            
            l=int(data[0].strip().split(";",1)[0].strip('\''))
            filebase=os.path.basename(file)[:-8]
            ture_label=l
            l_len=int(fea_len*config.label_repeat_factor) # k*r

            label=[]
            if config.label_add_blank:
                for i in range(l_len):
                    label.extend([config.class_num-2,l])
                label.append(config.class_num-2)
                l_len=l_len*2+1
            else:
                label=[l for i in range(l_len)]  # [A A A A ]

            # add fea data
            for temp in data:
                temp=temp.strip().split(";")[2:]
                fea.append(temp)
            fea=np.array(fea,dtype=np.float)
            
            # calculate  mean, variance
            mu = np.average(fea, axis=0)
            var = np.var(fea, axis=0)
            old_mu=self.mu
            self.mu=self.mu+(mu-self.mu)*fea_len/(self.frame_num+fea_len)
            self.var= float(self.frame_num)/(self.frame_num+fea_len)*(self.var+old_mu*old_mu)+float(fea_len)/(self.frame_num+fea_len)*(var+mu*mu)-self.mu*self.mu
            
            #add sample
            pack.append(fea)# feature seq
            pack.append(fea_len)#seq length
            pack.append(label)#label seq
            pack.append(l_len)#label length
            pack.append(ture_label)#true label
            self.sample_num=self.sample_num+1 # 5531
            self.data[file]=pack
            self.sample_names.append(file)        
            self.frame_num=self.frame_num+fea_len
    
    def add_specgram(self, filelist):
        with open(filelist, 'r') as fin:
            files = fin.readlines()
        for file in files:
            file = file.strip()
            label_key = file.split('_')[-1].split('.')[0]
            label = emo_class[label_key]
            pack=[]
            fea=[]
            
            # fea 
            fea = mpimg.imread(file) # [freq, time_step, alpha]
            fea = fea.swapaxes(1, 0) # [time_step, freq, alpha]
            fea = fea[:,:,0:3]
            
            fea_len = fea.shape[0]
            label_len = int(fea_len*config.label_repeat_factor)
            label_list =[label for i in range(label_len)]
            
            fea_reshape = np.reshape(fea, (-1,3))
            mu = np.average(fea_reshape, axis=0)
            var = np.var(fea_reshape, axis=0)
            old_mu_spec = self.mu_spec
            self.mu_spec = self.mu_spec+(mu-self.mu_spec)*fea_len*fea.shape[1]/(self.frame_num+fea_len*fea.shape[1])
            self.var_spec= float(self.frame_num)/(self.frame_num+fea_len*fea.shape[1])*(self.var_spec+old_mu_spec*old_mu_spec)+float(fea_len*fea.shape[1])/(self.frame_num+fea_len*fea.shape[1])*(var+mu*mu)-self.mu_spec*self.mu_spec
            
            # add sample
            pack.append(fea)
            pack.append(fea_len)
            pack.append(label_list) # label seq
            pack.append(label_len) # label length
            pack.append(label)
            self.sample_num=self.sample_num+1
            self.data[file]=pack
            self.sample_names.append(file)
            self.frame_num=self.frame_num+fea_len*fea.shape[1]
            
    def _shuffle(self):
        self.sample_heap = random.sample(self.sample_names,self.sample_num)
        
    def next_batch(self, batch_size):
        if self.batch_id == self.sample_num:
            self.batch_id = 0
            return [],[],[],[],[], 0
        if (self.batch_id == 0):
            if self.is_shuffle == True:
                self._shuffle()
        
        end_id = min(self.batch_id + batch_size, self.sample_num)
        key_batch=self.sample_heap[self.batch_id:end_id]

        fea_batch,fea_len_batch,label_batch,label_len_batch=self.prepare_batch(key_batch)


        batch_num = end_id-self.batch_id
        self.batch_id = end_id
        
        return key_batch,fea_batch,fea_len_batch,label_batch,label_len_batch,batch_num
        
    def prepare_batch(self,key_batch):
        fea_batch=[]
        fea_len_batch=[]
        label_batch=[]
        label_len_batch=[]
        for name in key_batch:
            fea_len_batch.append(self.data[name][1])
            label_len_batch.append(self.data[name][3])
        max_timesteps=np.max(fea_len_batch)
        max_label_len=np.max(label_len_batch)
        for i in range(len(key_batch)):
            #padding feature maxtrix
            temp=self.data[key_batch[i]][0]
            pad_n=max_timesteps-fea_len_batch[i]
            temp=np.pad(temp,[(0,pad_n),(0,0)],'constant')
            fea_batch.append(temp)
            #padding label
            temp=self.data[key_batch[i]][2]
            pad_n=max_label_len-label_len_batch[i]
            temp=np.pad(temp,[(0,pad_n)],'constant',constant_values=config.class_num)
            label_batch.append(temp)
        return fea_batch,fea_len_batch,label_batch,label_len_batch
        
    def next_batch_onehot(self, batch_size):
        if self.batch_id == self.sample_num:
            self.batch_id = 0
            return [],[],[],[],[],0
        if (self.batch_id == 0):
            if self.is_shuffle == True:
                self._shuffle()
         
        end_id = min(self.batch_id + batch_size, self.sample_num)
        key_batch=self.sample_heap[self.batch_id:end_id]
        batch_num = end_id-self.batch_id
        fea_batch,fea_len_batch,label_batch,label_len_batch,rectangle_num_batch, label_batch_true_label, label_batch_sequence=self.prepare_batch_onehot(key_batch)
        self.batch_id = end_id
        
        return key_batch,fea_batch,fea_len_batch,label_batch,label_len_batch,batch_num #,rectangle_num_batch, label_batch_true_label,label_batch_sequence
        
    def prepare_batch_onehot(self,key_batch):
        label_batch_sequence= []
        label_batch_true_label = []
        fea_batch=[]
        fea_len_batch=[]
        label_batch=[]
        label_len_batch=[]
        rectangle_num_batch = []
        
        for name in key_batch:
            fea_len_batch.append(self.data[name][1])
            label_len_batch.append(self.data[name][3])
            #label_len_batch.append(1)
        max_timesteps=np.max(fea_len_batch)
        max_label_len=np.max(label_len_batch)
        for i in range(len(key_batch)):
            #padding feature maxtrix
            temp=self.data[key_batch[i]][0]
            pad_n=max_timesteps-fea_len_batch[i]
            temp=np.pad(temp,[(0,pad_n),(0,0)],'constant')
            fea_batch.append(temp)
            #padding label
            l=self.data[key_batch[i]][4]
            l_oh=np.zeros(config.emo_num)
            l_oh[l] = 1
            label_batch.append(l_oh)
            label_batch_true_label.append([l])
            
            #padding label           
            temp=self.data[key_batch[i]][2]
            pad_n=max_label_len-label_len_batch[i]
            temp=np.pad(temp,[(0,pad_n)],'constant',constant_values=config.class_num)
            label_batch_sequence.append(temp)
            
        return fea_batch,fea_len_batch,label_batch,label_len_batch,rectangle_num_batch,label_batch_true_label,label_batch_sequence

    def prepare_batch_onehot_split(self,key_batch):
        fea_len_batch=[]
        label_batch=[]
        label_len_batch=[]
        wav_fea_batch=[]
        rectangle_num_batch=[]
        for name in key_batch:
            fea_len_batch.append(self.data[name][1])
            label_len_batch.append(self.data[name][3])
        
        max_timesteps=np.max(fea_len_batch)
        max_label_len=np.max(label_len_batch)
        
        fea_len_batch = []
        for i in range(len(key_batch)):
            if config.context:
                l=self.data[key_batch[i]][4]
                l_oh=np.zeros(config.emo_num)
                l_oh[l]=1
                label_batch.append(l_oh)
                
                print('hello333')
                fea_batch = []
                rectangle_16_num = 0 
                temp=self.data[key_batch[i]][0]
                fea_len = self.data[key_batch[i]][1]
                if fea_len <= 15:
                    print('fea_len < 15, prepare batch features error!!!')
                    exit()
                for j in range(fea_len-15):
                    fea_batch.append(temp[0+j:16+j, :])
                    fea_len_batch.append(16)
                    rectangle_16_num = rectangle_16_num+1
                for m in range(max_timesteps-fea_len):
                    fea_batch.append(np.zeros([16,config.fea_dim]))
                    fea_len_batch.append(16)
                wav_fea_batch.append(fea_batch)
                rectangle_num_batch.append(rectangle_16_num)
            
        return np.array(wav_fea_batch),np.array(fea_len_batch),np.array(label_batch),np.array(label_len_batch),np.array(rectangle_num_batch)
        
    def next_batch_onehot_spec(self, batch_size):
        if self.batch_id == self.sample_num:
            self.batch_id = 0
            return [],[],[],[],[],0
        if (self.batch_id == 0):
            if self.is_shuffle == True:
                self._shuffle()
        
        end_id = min(self.batch_id+batch_size,self.sample_num)
        key_batch=self.sample_heap[self.batch_id:end_id]
        batch_num=end_id-self.batch_id
        
        fea_batch, fea_len_batch, label_batch, label_len_batch = self.prepare_batch_onehot_spec(key_batch)
        self.batch_id=end_id
        
        return key_batch, fea_batch, fea_len_batch, label_batch, label_len_batch,batch_num
    
    def prepare_batch_onehot_spec(self, key_batch):
        fea_batch=[]
        fea_len_batch=[]
        label_batch=[]
        label_len_batch=[]
        
        for name in key_batch:
            fea_len_batch.append(self.data[name][1])
            label_len_batch.append(self.data[name][3])
            
        max_timesteps = np.max(fea_len_batch)
        max_label_len = np.max(label_len_batch)
        for i in range(len(key_batch)):
            # padding fea maxtrix
            temp=self.data[key_batch[i]][0]
            pad_n=max_timesteps-fea_len_batch[i]   # time_step , freq, alpha
            temp=np.pad(temp, [(0,pad_n), (0, 0), (0,0)], 'constant')
            fea_batch.append(temp)
            #padding label
            l=self.data[key_batch[i]][4]
            l_oh = np.zeros(config.emo_num)
            l_oh[l] = 1
            label_batch.append(l_oh)
        
        return fea_batch, fea_len_batch, label_batch, label_len_batch


