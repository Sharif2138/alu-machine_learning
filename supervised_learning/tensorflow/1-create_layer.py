#!/usr/bin/env python3
"""
1-create_layer.py
Creates a fully connected neural network layer
"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Creates a layer of a neural network using He initialization

    Args:
        prev: tensor output of previous layer
        n: number of nodes in the layer
        activation: activation function

    Returns:
        The tensor output of the layer
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=initializer,
                            name='layer')
    return layer(prev)
