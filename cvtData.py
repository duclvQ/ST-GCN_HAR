import os 
import numpy as np
import tqdm as tqdm
class cvtData:
    def cvt_Data():
        if not os.path.exists('./fphab_data/newData'):
            os.mkdir('./fphab_data/newData')
            #for i in range(1,7):
            #    os.mkdir('./fphab_data/newData/Subject_'+str(i))
        subject_list=os.listdir('./fphab_data/data')
        for s in subject_list:
            action_list=os.listdir('./fphab_data/data/'+s)
            index=0
            for a in action_list:
                index+=1
                number=os.listdir('./fphab_data/data/'+s+'/'+a)
                datas=[]
                for i in number:
                    lines= open('./fphab_data/data/'+s+'/'+a+'/'+i+'/skeleton.txt','rt').read().strip().split('\n')
                    f=open('./fphab_data/newData/'+str(index)+'_'+s+'_'+i+'.txt','w')
                    for l in lines:
                        frame_data =  l.strip().split(' ')
                        frame_data = frame_data[1:]
                        count=1
                        for fr in frame_data:
                            fr=str(fr)
                            #f.write(fr)
                            f.write(fr+' ')
                            if count%3==0:
                                f.write('\n')
                            count+=1

                        #frame_data.reshape(21,3)
                        #listToString = (' '.join(str(elm) for elm in frame_data))
                        #f.write('\n')
                        #f.write(listToString)
                        
                        




cvtData.cvt_Data()
print('done')
        
        