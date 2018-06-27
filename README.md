# 2018
## Summary ##



### Data ###

####- Databases:



1) IEMOCAP 



2) eNTERFACE 



3) MEC2016/2017 



4) EmotiW 2017/2018 



5) Berlin Corpus



6) Semaine



####- Papers

[1]Wenjing Han, Haifeng Li, Huabin Ruan, Lin Ma, Jiayin Sun, Bjoern Schuller, Active Learning for Dimensional Speech Emotion Recognition, Interspeech, 2013



[2]Wenjing Han, Huabin Ruan, Xiaojie Yu, Xuan Zhu, Combining feature selection and representation for speech emotion recognition, workshop of ICME, 2016



[3]Wenjing Han, Eduardo Coutinho, Huabin Ruan*, Haifeng Li, Bjoern Schuller, Xiaojie Yu, Xuan Zhu. Semi-Supervised Active Learning for Sound Classification in Hybrid Learning Environments[J]. Plos One, 2016, 11(9):e0162075.



[4]Wenjing Han, Huabin Ruan, Chao Wang, Tao Yang, Xuan Zhu, ACGAN based Semi-supervised Facial Expression Recognition, ACMMC workshop of Interspeech, 2017



[5]Xiaohan Xia, Jiamu Liu, Wenjing Han, Xuan Zhu, Hichem Sahli, Dongmei Jiang, Speech Emotion Recognition Based on Global Features and DCNN, ACMMC workshop of Interspeech, 2017



[6] Le Yang, Dongmei Jiang, Wenjing Han, Hichem Sahli, DCNN and DNN Based Multi-modal Depression Recognition, ACII, 2017



[7]Xiaomin Chen, Wenjing Han, Huabin Ruan, Jiamu Liu, Haifeng Li, Dongmei Jiang, Sequence-to-sequence Modelling for Categorical Speech Emotion Recognition Using Recurrent Neural Network, ACII Asia, Beijing, 2018



[8]Taorui Ren, Huabin Ruan, Wenjing Han, Tao Yang, Dongmei Jiang, Video-based Emotion Recognition Using Multi-dichotomy RNN-DNN, ACII Asia, Beijing, 2018



[9]Jiamu Liu, Wenjing Han, Huabin Ruan, Xiaomin Chen, Dongmei Jiang, Haifeng Li, Learning Salient Features for Speech Emotion Recognition Using CNN, ACII Asia, Beijing, 2018



[10]Xiaohan Xia, Jiamu Liu, Tao Yang, Dongmei Jiang, Wenjing Han, Hichem Sahli, Video Emotion Recognition using Hand-Crafted and Deep Learning Features, ACII Asia, Beijing, 2018



[11]Wenjing Han, Huabin Ruan, Xiaomin Chen, Zhixiang Wang, Haifeng Li, Bjoern Schuller, Towards Temporal Modelling of Categorical Speech Emotion Recognition,  Interspeech, Hyderabad, 2018



####- Scripts



1) Attention



2) CTC



3) CNN





###- Tools



####1) Python Anaconda



####2) Tensorflow



#####Install requirement: 



https://pypi.org/search



https://www.lfd.uci.edu/~gohlke/pythonlibs/#tensorflow



遇到的问题：1）html5lib0.9999999(七个9)安装后，pip list看到该版本，list中仍旧是1.0版本。这是因为需要先将高版本删除。

2）tensorboard的使用："tansorboard --logdir=graphs2"出现“tensorboard No dashboards are active for the current data set.”的报错。注意路径没用引号，且注意路径是否正确。



#####Learning Materials:



知乎合集：

https://www.zhihu.com/question/49909565



机器学习google课程：

https://developers.google.cn/machine-learning/crash-course/prereqs-and-prework



斯坦福C20课件：

http://web.stanford.edu/class/cs20si/syllabus.html

https://github.com/chiphuyen/stanford-tensorflow-tutorials/



#####github rescource:



音频事件分类Audioset VGGish: 

https://github.com/tensorflow/models/tree/master/research/audioset



3) Opensmile



4) Kaldi



5) MarkdownPad



- Subjects



1) Speech emotion recognition



2) Keyword spotting



3) Sound classification





### Knowledge and Skills###



#### - logistic regression 和 linear regression最好的解释

浅谈Logistic回归及过拟合 https://www.cnblogs.com/hxsyl/p/4922780.html



Key content: LogisticRegression 就是一个被logistic方程归一化后的线性回归，仅此而已。



#### - dropout， ReLU 

dropout：一般加在全连接层的后一两层，但是batchnorm的paper说，用了batchnorm后就可以不用dropout了。dropout可以减轻过拟合状态。dropout的比率keep_prob在训练和测试阶段一般设置为不同值，训练时小于1，测试时等于1。



ReLU： 相比sigmoid容易导致梯度弥散，就是说梯度值随着传递会指数级减小。因此会导致最后几层网络的权值不能得到有效的更新。ReLU可以很好地传递梯度，经过多层传播，梯度仍旧不会大幅缩水，因此非常适合训练很深的网络。ReLU层正面解决了梯度弥散的问题，而不需要通过DBN那样的无监督逐层初始化来曲线救国。目前的CNN隐含层一般使用ReLU，因为层数很深，但RNN中的隐层多使用Tanh，但输出层还是要用sigmoid，因为sigmoid最能表达概率的含义。
