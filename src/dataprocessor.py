import torch
import numpy as np

class Data_transform():
    def __init__(self, data_transform=True):
        self.data_transform = data_transform

    def __call__(self, x):
        if self.data_transform:
            C, T, V, M = x.shape
            x_new = np.zeros((C*3, T, V, M))
            x_new[:C,:,:,:] = x
            for i in range(T-1):
                x_new[C:(2*C),i,:,:] = x[:,i+1,:,:] - x[:,i,:,:]
            for i in range(V):
                x_new[(2*C):,:,i,:] = x[:,:,i,:] - x[:,:,0,:]
            return x_new
        else:
            return x

class Occlusion_part():
    def __init__(self, occlusion_part=[]):
        self.occlusion_part = occlusion_part
        self.parts = dict()
        self.parts[1] = np.array([1, 2,  7,  8,  9]) - 1  # Thumb
        self.parts[2] = np.array([1, 3, 10, 11, 12]) - 1  # Index
        self.parts[3] = np.array([1, 4, 13, 14, 15]) - 1  # Middle
        self.parts[4] = np.array([1, 5, 16, 17, 18]) - 1  # Ring
        self.parts[5] = np.array([1, 6, 19, 20, 21]) - 1  # Pinky

    def __call__(self, x):
        for part in self.occlusion_part:
            x[:,:,self.parts[part],:] = 0
        return x


class Occlusion_time():
    def __init__(self, occlusion_time=0):
        self.occlusion_time = int(occlusion_time // 2)

    def __call__(self, x):
        if not self.occlusion_time == 0:
            x[:,(50-self.occlusion_time):(50+self.occlusion_time),:,:] = 0
        return x
