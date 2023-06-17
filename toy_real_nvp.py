import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

class RealNVP(tf.keras.Model):
    def __init__(self, num_layers, input_dim):
        super().__init__()
        self.L = num_layers
        self.D = input_dim
        self.mask_list = [self.create_mask(flip=(i%2)) for i in range(self.L)]
        self.s_list = [self.create_s() for _ in range(self.L)]
        self.t_list = [self.create_t() for _ in range(self.L)]
        self.logdetJ = None

        #proposal distribution
        self.Rz = tfp.distributions.MultivariateNormalDiag(loc = np.zeros(self.D), 
                                                           scale_diag = np.ones(self.D))
    
    def create_s(self):
        s = Sequential(
            Dense(64, activation='relu', kernel_regularizer = l2(0.01)),
            Dense(64, activation='relu', kernel_regularizer = l2(0.01)),
            Dense(self.D, activation='relu', kernel_regularizer = l2(0.01))    
        )
        return s
    
    def create_t(self):
        t = Sequential(
            Dense(64, activation='relu', kernel_regularizer = l2(0.01)),
            Dense(64, activation='relu', kernel_regularizer = l2(0.01)),
            Dense(self.D, activation='relu', kernel_regularizer = l2(0.01))      
        )
        return t
    
    def create_mask(self,flip=False):
        mask = tf.concat([np.ones((self.D//2,1)),np.zeros((self.D-self.D//2,1))],axis=0)
        if flip:
            mask = tf.reverse(mask, [0])
        return mask

    def forward(self, z):
        self.logdetJ = 0
        for i in range(self.L):
            mask = self.mask_list[i]
            z_ = z*mask
            s = self.list[i](z_)
            t = self.t_list[i](z_)
            x = z_ + tf.exp(s)*z*(1-mask) + t
            self.logdetJ += s
            z = x
        return x

    def backward(self, x):
        for i in range(self.L-1,0,-1):
            mask = self.mask_list[i]
            x_ = x*mask
            s = self.s_list[i](x_)
            t = self.t_list[i](x_)
            z = x_ + (x*(1-mask) - t)*tf.exp(-s)
            x = z
        return z

if __name__ == '__main__':
    # Dataset:
    data = make_moons(3000, noise=0.05)[0].astype('float32')
    norm = tf.keras.layers.Normalization()
    norm.adapt(data)
    norm_data = norm(data)
    print(norm_data.shape, norm_data[0])
    
    # plt.scatter(norm_data[:,0], norm_data[:,1])
    # plt.show()


    model = RealNVP(2, 4)
    


