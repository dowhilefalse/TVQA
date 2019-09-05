# __author__ = "Jie Lei"

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
import tensorflow as tf

class BidafAttn(nn.Module):
    """from the BiDAF paper https://arxiv.org/abs/1611.01603.
    Implemented by @easonnie and @jayleicn
    """
    def __init__(self, channel_size, method="original", get_h=False):
        super(BidafAttn, self).__init__()
        """
        This method do biDaf from s2 to s1:
            The return value will have the same size as s1.
        :param channel_size: Hidden size of the input
        """
        self.channel_size = channel_size * 3
        self.method = method
        self.get_h = get_h
        
    def dense(inputs):
        return tf.layers.dense(inputs,
                    self.channel_size * 3,
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
                    reuse=None)
    
    def similarity(self, s1, l1, s2, l2):
        """
        :param s1: [B, t1, D]
        :param l1: [B]
        :param s2: [B, t2, D]
        :param l2: [B]
        :return:
        """
        if self.method == "original":
            t1 = s1.shape[1]
            t2 = s2.shape[1]
            repeat_s1 = tf.tile(tf.expand_dims(t1, 2), [1, 1, t2, 1])
            repeat_s2 = tf.tile(tf.expand_dims(t1, 2), [1, t1, 1, 1])
            packed_s1_s2 = tf.concat([repeat_s1, repeat_s2, repeat_s1 * repeat_s2], 3)
            s = self.dense(packed_s1_s2)
#             repeat_s1 = s1.unsqueeze(2).repeat(1, 1, t2, 1)  # [B, T1, T2, D]
#             repeat_s2 = s2.unsqueeze(1).repeat(1, t1, 1, 1)  # [B, T1, T2, D]

#             packed_s1_s2 = torch.cat([repeat_s1, repeat_s2, repeat_s1 * repeat_s2], dim=3)  # [B, T1, T2, D*3]
#             s = self.mlp(packed_s1_s2).squeeze()  # s is the similarity matrix from biDAF paper. [B, T1, T2]
        elif self.method == "dot":
            s = tf.matmul(s1, tf.transpose(s2, [1, 2]))
#             s = torch.bmm(s1, s2.transpose(1, 2))

        
#         s_mask = s.data.new(*s.size()).fill_(1).byte()  # [B, T1, T2]
        s_mask = tf.Variable(tf.ones_like(s))
        # Init similarity mask using lengths
        for i, (l_1, l_2) in enumerate(zip(l1, l2)):
            tf.assign(s_mask[i][:l_1, :l_2], 0)
        
#         s.data.masked_fill_(s_mask.data.byte(), -float("inf"))
        return s

    @classmethod
    def get_u_tile(cls, s, s2):
        """
        attended vectors of s2 for each word in s1,
        signify which words in s2 are most relevant to words in s1
        """
        a_weight = tf.softmax(s, 2)  # [B, t1, t2]
        tf.assign(a_weight[tf.where(tf.is_nan(a_weight))], 0)
#         a_weight.data.masked_fill_(a_weight.data != a_weight.data, 0)  # remove nan from softmax on -inf
#         u_tile = torch.bmm(a_weight, s2)  # [B, t1, t2] * [B, t2, D] -> [B, t1, D]
        u_til = tf.matmul(a_weight, s2)
        a_weight = F.softmax(s, dim=2)  # [B, t1, t2]
        tf.assign(a_weight[tf.where(tf.is_nan(a_weight))], 0)
#         a_weight.data.masked_fill_(a_weight.data != a_weight.data, 0)  # remove nan from softmax on -inf
#         u_tile = torch.bmm(a_weight, s2)  # [B, t1, t2] * [B, t2, D] -> [B, t1, D]
        u_til = tf.matmul(a_weight, s2)
        return u_tile

    @classmethod
    def get_h_tile(cls, s, s1):
        """
        attended vectors of s1
        which words in s1 is most similar to each words in s2
        """
        t1 = s1.shape[1]
        b_weight = tf.reshape(tf.softmax(tf.max(s, 2)[0], -1),[-1,1])
        h_tile = tf.tile(tf.matmul(b_weight, s1), [1, t1, 1])
#         b_weight = F.softmax(torch.max(s, dim=2)[0], dim=-1).unsqueeze(1)  # [b, t2]
#         h_tile = torch.bmm(b_weight, s1).repeat(1, t1, 1)  # repeat to match s1 # [B, t1, D]
        return h_tile

    def forward(self, s1, l1, s2, l2):
        s = self.similarity(s1, l1, s2, l2)
        u_tile = self.get_u_tile(s, s2)
        # h_tile = self.get_h_tile(s, s1)
        h_tile = self.get_h_tile(s, s1) if self.get_h else None
        return u_tile, h_tile
        # return u_tile

