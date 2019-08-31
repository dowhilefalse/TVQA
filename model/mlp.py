__author__ = "Jie Lei"

import torch
import torch.nn as nn
import tensorflow as tf

class MLP():
    def __init__(self, in_dim, out_dim, hsz, n_layers):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hsz = hsz
        self.layers = inputs
        
    def dense(inputs):
        return tf.layers.dense(
            inputs,
            self.hsz,
            activation=None,
            use_bias=True,
            kernel_initializer=None,
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=True,
            name=None,
            reuse=None
        )
    
    def forward(self, layers):
        for i in range(n_layers):
            if i == n_layers - 1:
                layers = self.dense(self.out_dim) 
#                 layers.append(nn.Linear(prev_dim, out_dim))
            else:
                layers = self.dense(layers, self.hsz)
                layers = tf.nn.relu(layers)
                layers = tf.nn.dropout(layers, 0.5)
        return layers

if __name__ == '__main__':
    test_in = torch.randn(10, 300)

    mlp1 = MLP(300, 1, 100, 1)
    print("="*20)
    print(mlp1)
    print(mlp1(test_in).size())

    mlp2 = MLP(300, 10, 100, 2)
    print("="*20)
    print(mlp2)
    print(mlp2(test_in).size())

    mlp3 = MLP(300, 5, 100, 4)
    print("=" * 20)
    print(mlp3)
    print(mlp3(test_in).size())
