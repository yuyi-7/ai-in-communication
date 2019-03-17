import tensorflow as tf
import CNN_interface
import DNN_interface


INPUT_NODE = 64  # 输入节点
OUTPUT_NODE = 64  # 输出节点

sent_data_shape = None  # 发射机的输出维度
sent_dnn_DROP = 0.5  # 发射机的DNN的drop
sent_dnn_REGULARIZER_RATE = 1e-4  # 发射机的DNN的正则率

receive_data_after_cnn_shape = 64  # 接收机通过CNN网络之后的数据维度
receive_cnn_DROP = 0.5  # 接收机的CNN的drop，CNN内的一个密集层的drop
receive_cnn_REGULARIZER_RATE = 1e-4  # 接收机中CNN的密集层的正则率

receive_data_shape = None  # 接收机的输出维度
receive_dnn_DROP = 0.5  # 接收机的DNN的drop
receive_dnn_REGULARIZER_RATE = 1e-4  # 接收机的DNN的正则率

LEARNING_RATE_BASE = 0.8 # 模型基础学习速率
LEARNING_RATE_DECAY = 0.99  # 学习衰减速度
BATCH_SIZE = 200  # 一批数据量
TRAIN_NUM = 20000  # 数据总量
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减
TRAINING_STEPS = 10000  # 训练多少次

SNR = 0   # 信噪比

E_x = 10 ** (0.1*SNR)  #信号能量

import numpy as np
import pandas as pd


# 生成数据
Y = np.random.randint(0,2,[TRAIN_NUM,OUTPUT_NODE]).astype('float32')
X = np.array(pd.DataFrame(Y).applymap(lambda x: 1 if x==1 else -1)).astype('float32')


# 定义整个模型的x和y
x = tf.placeholder(tf.float32 ,[None,INPUT_NODE], name='x_input')
y_ = tf.placeholder(tf.float32, [None,OUTPUT_NODE], name='y-input')

"""
# 发射机通过DNN输出的数据
# dnn_interface(input_tensor, output_shape, regularizer_rate=None, drop=None)
sent_data = DNN_interface.dnn_interface(input_tensor=x,
                                        output_shape=sent_data_shape,
                                        regularizer_rate=sent_dnn_REGULARIZER_RATE,
                                        drop=sent_dnn_DROP )
"""

# 过信道,加噪声
data_after_channle = X * E_x + np.random.randn(TRAIN_NUM,64)  # sigma * r + mu

# 接收机的CNN网络
# cnn_inference(input_tensor, output_shape, drop=None, regularizer_rate=None)
receive_data_after_cnn = CNN_interface.cnn_inference(input_tensor=data_after_channle,
                                                     output_shape=receive_data_after_cnn_shape,
                                                     drop=receive_cnn_DROP,
                                                     regularizer_rate=receive_cnn_REGULARIZER_RATE)

# 移除噪声
data_after_remove_voice = data_after_channle - receive_data_after_cnn

# 判断函数
def judge_cnn(data):
    mat = [1,-1]
    data = pd.DataFrame(data)
    data_after_judge = data.applymap(lambda x: mat[0] if np.sign(x) > 0 else mat[1])
    return np.array(data_after_judge)

data_judged = tf.py_func(judge_cnn, [data_after_remove_voice], tf.float32)


# 接收机的DNN
# dnn_interface(input_tensor, output_shape, regularizer_rate=None, drop=None)
y = DNN_interface.dnn_interface(input_tensor=data_judged,
                                output_shape=OUTPUT_NODE,
                                regularizer_rate=receive_dnn_REGULARIZER_RATE,
                                drop=receive_dnn_DROP,
                                )

# DNN后的判断函数
def judge_dnn(data):
    mat = [1,0]
    data = pd.DataFrame(data)
    data_after_judge = data.applymap(lambda x: mat[0] if np.sign(x) > 0 else mat[1])
    return np.array(data_after_judge)

y_judged = tf.py_func(judge_dnn, [y], tf.float32)

# 损失函数
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_judged,
                                                               labels=y_)  # 自动one-hot编码
cross_entropy_mean = tf.reduce_mean(cross_entropy)  # 平均交叉熵

loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))  # 损失函数是交叉熵和正则化的和

# 优化器
# 定义当前迭代轮数的变量
global_step = tf.get_variable('global_step',  # 存储当前迭代的轮数
                              dtype=tf.int32,  # 整数
                              initializer=0,  # 初始化值
                              trainable=False)  # 不可训练
# 定义学习速率
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,  # 基础学习率
                                           global_step,  # 当前迭代轮数
                                           TRAIN_NUM / BATCH_SIZE,  # 迭代次数
                                           LEARNING_RATE_DECAY,  # 学习衰减速度
                                           staircase=False)  # 是否每步都改变速率

# 定义优化函数
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)

"""
# 滑动平均类
variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
variables_averages_op = variable_averages.apply(tf.trainable_variables())
train_op = tf.group(train_step, variables_averages_op)
"""

# 保存模型
saver = tf.train.Saver(max_to_keep=3)
min_loss = float('inf')

#使用tensorboard
#writer = tf.summary.FileWriter('/logs/log', tf.get_default_graph())
#writer.close()
merged = tf.summary.merge_all()

with tf.Session() as sess:
    #初始化写日志的writer,并将当前Tensorflow计算图写入日志
    summary_writer = tf.summary.FileWriter('/logs/log', sess.graph)
    tf.global_variables_initializer().run()  # 初始化
    for i in range(TRAINING_STEPS):
        # 设置批次
        start = (i * BATCH_SIZE) % TRAIN_NUM
        end = min(start+BATCH_SIZE, TRAIN_NUM)
        loss_entropy = sess.run(cross_entropy_mean,
                                feed_dict={x: X[start:end], y_: Y[start:end]})
        compute_loss,summary = sess.run([train_step, merged],
                                feed_dict={x:X[start:end], y_:Y[start:end]})
        # 保存模型
        if compute_loss < min_loss:
            min_loss = compute_loss
            saver.save(sess, '/ckpt/min_loss_model.ckpt', global_step=i)

        # 写入tensorboard日志
        summary_writer.add_summary(summary,i)

        # 输出
        if i % 100:
            print('训练了%d次后,交叉熵损失为%f,总损失为%f'%(i,loss_entropy,compute_loss))

sess.close()