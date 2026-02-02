#!/usr/bin/env python3
"""
0-create_placeholders.py
Creates placeholders for input data and labels
"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Creates two placeholders for a neural network

    Args:
        nx: number of input features
        classes: number of classes

    Returns:
        x: placeholder for input data
        y: placeholder for one-hot labels
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    return x, y
