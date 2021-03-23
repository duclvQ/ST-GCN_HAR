import numpy as np
import pickle
from tqdm import tqdm

class get_test_samples():
    def __init__(self,run=1,type='predicted'):
        self.run=run
        self.type=type
        self.dir_1 ="/media/data3/duc/F-PHAB/FPHAB/data_Nam/FPHAB_hand3d_run1_Y2Y.pkl"
        self.dir_2 ="/media/data3/duc/F-PHAB/FPHAB/data_Nam/FPHAB_hand3d_run2_X2Y.pkl"
        self.label_dir="/media/data3/duc/F-PHAB/FPHAB/label_fphab.txt"
    def __getitem__(self,idx):
        test_samples , _= self.get_data()
        if self.type== 'fixed':
            _ , test_samples= self.get_data()
        


    def get_data(self):
        if self.run==1:
            filepkl=open(self.dir_1, 'rb')
        else:
            filepkl=open(self.dir_2,'rb')

        data=pickle.load(filepkl)
        #test_list=[None]*270
        label_list = open(label_dir,'rt').readlines()
        #print(len(label_list))
        z=np.zeros((21,3),dtype='float')
        predicted_list=[[z for _ in range(100)] for _ in range(270) ]
        fixed_list=[[z for _ in range(100)] for _ in range(270) ]
        for frame in tqdm(data):
            subject= frame['path'].split('/')[7]
            subject=int(subject.split('_')[-1])
            #print(type(subject))
            seq= int(frame['path'].split('/')[9])
            frame_idx=(((frame['path'].split('/')[-1]).split('.'))[0]).split('_')[-1]
            frame_idx=int(frame_idx)
            if frame_idx>=100:
                continue
            #print((frame_idx))
            action_name=frame['path'].split('/')[8]
            idx=0
            for i,action in enumerate(label_list):
            if action==action_name:
                idx=i
                break 
        
            predicted_list[(subject-1)*45+idx][frame_idx]= frame['predicted']
            fixed_list[(subject-1)*45+idx][frame_idx]=frame['fixed']
        #print(predicted_list[0])
        return predicted_list, fixed_list