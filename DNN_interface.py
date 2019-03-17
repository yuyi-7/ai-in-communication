import tensorflow as tf


LAYER1_NODE = 512  # 第一层节点

LAYER2_NODE = 1024  # 第二层节点

LAYER3_NODE = 512  # 第三层节点

"""
input layer

layer1 512  relu

layer2 1024  relu

layer3 512  relu

output layer
"""


def dnn_interface(input_tensor, output_shape, regularizer_rate=None, drop=None):

    # 第一层密集层，添加L2正则
    with tf.variable_scope('layer1'):
        layer1_weight = tf.get_variable('weight', [input_tensor._shape_tuple()[1], LAYER1_NODE],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))

        if regularizer_rate != None:
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer_rate)(layer1_weight))

        layer1_biase = tf.get_variable('biase', [LAYER1_NODE],
                                       initializer=tf.constant_initializer(0.0))

        layer1 = tf.nn.relu(tf.matmul(input_tensor, layer1_weight) + layer1_biase)

        if drop != None:
            layer1 = tf.nn.dropout(layer1, drop)

    # 第二层密集层，添加L2正则
    with tf.variable_scope('layer2'):
        layer2_weight = tf.get_variable('weight', [LAYER1_NODE, LAYER2_NODE],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))

        if regularizer_rate != None:
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer_rate)(layer2_weight))

        layer2_biase = tf.get_variable('biase', [LAYER2_NODE],
                                       initializer=tf.constant_initializer(0.0))

        layer2 = tf.nn.relu(tf.matmul(layer1, layer2_weight) + layer2_biase)

        if drop != None:
            layer2 = tf.nn.dropout(layer2, drop)

    # 第三层密集层，添加L2正则
    with tf.variable_scope('layer3'):
        layer3_weight = tf.get_variable('weight', [LAYER2_NODE, LAYER3_NODE],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer_rate != None:
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer_rate)(layer3_weight))

        layer3_biase = tf.get_variable('biase', [LAYER3_NODE],
                                       initializer=tf.constant_initializer(0.0))

        layer3 = tf.nn.relu(tf.matmul(layer2, layer3_weight) + layer3_biase)

        if drop != None:
            layer3 = tf.nn.dropout(layer3, drop)

    # 第四层，输出层
    with tf.variable_scope('layer4_output'):
        layer4_weight = tf.get_variable('weight', [LAYER3_NODE, output_shape],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer_rate != None:
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer_rate)(layer4_weight))

        layer4_biase = tf.get_variable('biase', [output_shape],
                                       initializer=tf.constant_initializer(0.0))

        layer4 = tf.matmul(layer3, layer4_weight) + layer4_biase

    return layer4
