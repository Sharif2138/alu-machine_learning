#!/usr/bin/env python3
"""
5-create_train_op.py
Creates the training operation using gradient descent
"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Creates the training operation

    Args:
        loss: loss tensor
        alpha: learning rate

    Returns:
        training operation tensor
    """
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train_op = optimizer.minimize(loss)
    return train_op
