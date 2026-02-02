#!/usr/bin/env python3
"""
2-forward_prop.py
Performs forward propagation for the neural network
"""

import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph

    Args:
        x: placeholder for input data
        layer_sizes: list of nodes in each layer
        activations: list of activation functions

    Returns:
        y_pred: tensor output of the network
    """
    out = x
    for i in range(len(layer_sizes)):
        out = create_layer(out, layer_sizes[i], activations[i])
    return out
