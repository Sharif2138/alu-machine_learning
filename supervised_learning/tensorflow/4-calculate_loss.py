#!/usr/bin/env python3
"""
4-calculate_loss.py
Calculates the softmax cross-entropy loss
"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Calculates softmax cross-entropy loss

    Args:
        y: placeholder for true labels
        y_pred: predicted output tensor

    Returns:
        loss tensor
    """
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=y_pred, labels=y))
    return loss
