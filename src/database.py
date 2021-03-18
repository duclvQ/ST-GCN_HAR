import os
import os.path as osp
import yaml
import random
import torch
import numpy as np
from torch.utils.data import Dataset
import random
class FPHAB(Dataset):

	
    def __init__(self, path, type='train', subset='31', data_shape=(3,100,21,1), transform=None, fphab_model_type=None,fileName='10_s_2_1.txt'):
        self.path = path
        self.maxC, self.maxT, self.maxV, self.maxM = data_shape
        self.transform = transform
        self.subset=subset
        self.fileName=fileName
        self.fphab_model_type = fphab_model_type
        #self.type=type
        if not os.path.exists('./datasets/' + self.subset + '_' + type + '.txt'):
            self.get_train_list()
        
        fr = open('./datasets/' + self.subset + '_' + type + '.txt', 'r')
        self.files = fr.readlines()
        fr.close()

    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        file_name = self.files[idx].strip()
        label = file_name.split('/')[-1]
        label, _, _ , _ = label.split("_")
        label = int(label) - 1

        data = np.zeros((self.maxC, self.maxT, self.maxV, self.maxM))
        location = np.zeros((2, self.maxT, self.maxV, self.maxM))
        with open(self.path + file_name, 'r') as fr:
            skeletonData = fr.readlines()
            # data in format 'x\ty\tz' need to be split to format 'x,y,z'
            skeletonData = [item.split() for item in skeletonData]
            frame_num = int(len(skeletonData) / self.maxV)
            for frame in range(frame_num):
                if frame >= self.maxT:
                    break
                frameData = skeletonData[(frame*21):(frame+1)*21]
                hand_num = 1
                for hand in range(hand_num):
                    for joint in range(self.maxV):
                        data[0,frame,joint,hand] = float(frameData[joint][0])
                        data[1,frame,joint,hand] = float(frameData[joint][1])
                        data[2,frame,joint,hand] = float(frameData[joint][2])
                        location[0,frame,joint,hand] = 0
                        location[1,frame,joint,hand] = 0
                        
        if frame_num <= self.maxT:
            data = data[:,:self.maxT,:,:]
        if self.transform:
            data = self.transform(data)
        #print(data[:,:,:,0])
        data = torch.from_numpy(data).float()
        location = torch.from_numpy(location).float()
        label = torch.from_numpy(np.array(label)).long()
 
        return data, location, label, file_name
    '''
    def get_test_sample(self, fileName):
        file_name = self.fileName
        #label = file_name.split('/')[-1]
        label, _, _ , _ = label.split("_")
        label = int(label) - 1

        data = np.zeros((self.maxC, self.maxT, self.maxV, self.maxM))
        location = np.zeros((2, self.maxT, self.maxV, self.maxM))
        with open( file_name, 'r') as fr:
            skeletonData = fr.readlines()
            # data in format 'x\ty\tz' need to be split to format 'x,y,z'
            skeletonData = [item.split() for item in skeletonData]
            frame_num = int(len(skeletonData) / self.maxV)
            for frame in range(frame_num):
                if frame >= self.maxT:
                    break
                frameData = skeletonData[(frame*21):(frame+1)*21]
                hand_num = 1
                for hand in range(hand_num):
                    for joint in range(self.maxV):
                        data[0,frame,joint,hand] = float(frameData[joint][0])
                        data[1,frame,joint,hand] = float(frameData[joint][1])
                        data[2,frame,joint,hand] = float(frameData[joint][2])
                        location[0,frame,joint,hand] = 0
                        location[1,frame,joint,hand] = 0
                        
        if frame_num <= self.maxT:
            data = data[:,:self.maxT,:,:]
        if self.transform:
            data = self.transform(data)
        #print(data[:,:,:,0])
        data = torch.from_numpy(data).float()
        location = torch.from_numpy(location).float()
        label = torch.from_numpy(np.array(label)).long()
        return data, location, label, file_name
    '''        
    def get_train_list(self):
        folder = '/newData/'
        if not os.path.exists('./datasets'):
            os.mkdir('./datasets')

        files = [x for x in os.listdir(self.path + folder) if x.endswith(".txt")]
        random.shuffle(files)
        #print(len(files))
        test_dataset=[]
        train_dataset=[]
        if self.subset=='31':
          # 3:1 // train:test
          f_31_train = open('./datasets/31_train.txt','w')
          f_31_test = open('./datasets/31_test.txt','w')
          test_dataset = files[:int(len(files)/4)]
          train_dataset = files[int(len(files)/4):]
          for test in test_dataset:
              f_31_test.write(folder+test+'\n')
          for train in train_dataset:
              f_31_train.write(folder+train+'\n')
          f_31_train.close()
          f_31_test.close()
        elif self.subset=='13':
             # 1:3 // train:test
            f_13_train = open('./datasets/13_train.txt','w')
            f_13_test = open('./datasets/13_test.txt','w')
            test_dataset = files[:int(len(files)/4)*3]
            train_dataset = files[int(len(files)/4)*3:]
            for test in test_dataset:
              f_13_test.write(folder+test+'\n')
            for train in train_dataset:
              f_13_train.write(folder+train+'\n')
            f_13_train.close()
            f_13_test.close()

        elif self.subset=='t':
            train_id=[1,3,4,5,6,7]
            test_id=[2]
            test_sample_id=[8,9]
            f_t_train = open('./datasets/t_train.txt','w')
            f_t_test = open('./datasets/t_test.txt','w')
            f_t_test_sample = open('./datasets/t_test_sample.txt','w')
            for file in files:
                _, _,subject, i  = file.split("_")
                #print(i[0])
                if  int(i[0]) in train_id:
                    train_dataset=f_t_train.write(folder+file+'\n')
                elif int(i[0]) in test_sample_id:
                    test_sample_d=f_t_test_sample.write(folder+file+'\n')
                else:
                    test_dataset=f_t_test.write(folder+file+'\n')
                
            f_t_train.close()
            f_t_test.close()
            f_t_test_sample.close()
        else:
            train_id=[1,3,5,6,7]
            test_id=[2,4]
            f_11_train = open('./datasets/11_train.txt','w')
            f_11_test = open('./datasets/11_test.txt','w')
            for file in files:
                _, _,subject, i  = file.split("_")
                print(i[0])
                if  int(i[0]) in train_id:
                    train_dataset=f_11_train.write(folder+file+'\n')
                else:
                    test_dataset=f_11_test.write(folder+file+'\n')
                
            f_11_train.close()
            f_11_test.close()
        
        
        
        
class DatasetFactory():
    def __init__(self, path, type='train', subset='cs', data_shape=(3,100,21,1), transform=None, dataset_name='FPHAB', fphab_model_type=None):
        self.path = path
        self.type = type
        self.subset = subset
        self.data_shape = data_shape
        self.transform = transform
        self.dataset_name = dataset_name
        self.fphab_model_type = fphab_model_type
    def getDataset(self):
         
            return FPHAB(self.path, self.type, self.subset, self.data_shape, self.transform, self.fphab_model_type)

# support function
def countLineInFile(filePath):
    with open(filePath) as fp:
        lines = len(fp.readlines())
    return lines