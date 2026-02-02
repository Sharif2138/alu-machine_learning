#!/usr/bin/env python3
"""
3-calculate_accuracy.py
Calculates the accuracy of predictions
"""

import tensorflow as tf

def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction

    Args:
        y: placeholder for true labels
        y_pred: predicted output tensor

    Returns:
        accuracy tensor
    """
    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy
