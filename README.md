# Deep_Instance

目前分为两种，基于TensorFlow 和基于Pytorch

### TensorFlow

---

- Basic_LSTM
  - 描述：使用 [tf.Dataset](https://tensorflow.google.cn/api_docs/python/tf/data/Dataset) 方式进行数据迭代
  - 源码：[RNN_demo1](https://github.com/GYQ2017/Deep-Instance/blob/master/TensorFlow/RNN_demo1.py)

- Basic_LSTM
  - 描述：自己写数据迭代函数
  - 源码：[RNN_demo2](https://github.com/GYQ2017/Deep-Instance/blob/master/TensorFlow/RNN_demo2.py)

- BiGRU
  - 描述：使用 [tf.Dataset](https://tensorflow.google.cn/api_docs/python/tf/data/Dataset) 方式进行数据迭代，当使用train_test_split遇到ImportError,DLL load failed时,可参考[这里](http://blog.csdn.net/arthasking123/article/details/51762843) 
  - 源码：[RNN_demo3](https://github.com/GYQ2017/Deep-Instance/blob/master/TensorFlow/RNN_demo3.py) 

- linear
  - 描述：线性回归模型
  - 源码：[linear](https://github.com/GYQ2017/Deep-Instance/blob/master/TensorFlow/linear.py) 

### Pytorch

---

- __数组切片__ 
  - 描述：主要是针对数组的切片操作
  - 源码：[001.py](https://github.com/GYQ2017/Deep-Instance/blob/master/Pytorch/001.py) 
- __pytorch 计算图__ 
  - 描述：Pytorch 计算图和自动求导
  - 源码：[002.py](https://github.com/GYQ2017/Deep-Instance/blob/master/Pytorch/002.py) 
- __pytorch 映射__
  - 描述：深度学习构建模块:映射(Linear and non Linear)
  - 源码：[003.py](https://github.com/GYQ2017/Deep-Instance/blob/master/Pytorch/003.py) 
- __pytorch 简单实例__ 
  - 描述：简单实例
  - 源码：[004.py](https://github.com/GYQ2017/Deep-Instance/blob/master/Pytorch/004.py) 
- __pytorch 词嵌入__ 
  - 描述：2元N-Gram语言模型
  - 源码：[005.py](https://github.com/GYQ2017/Deep-Instance/blob/master/Pytorch/005.py) 
- __pytorch 词嵌入__ 
  - 描述：Cbow模型
  - 源码：[006.py](https://github.com/GYQ2017/Deep-Instance/blob/master/Pytorch/006.py) 
- __pytorch 序列模型__
  - 描述：LSTM网络
  - 源码：[007.py](https://github.com/GYQ2017/Deep-Instance/blob/master/Pytorch/007.py) 
