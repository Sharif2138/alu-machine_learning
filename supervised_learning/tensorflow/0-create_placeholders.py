#!/usr/bin/env python3
import tensorflow as tf

def create_placeholders(nx, classes):
    """
    Creates placeholders for input data and labels.

    nx: number of features
    classes: number of classes

    Returns: x, y placeholders
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y
