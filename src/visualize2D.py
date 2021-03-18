from matplotlib import animation as animation
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
import time
import matplotlib
#from IPython.display import HTML
import matplotlib.patches as patches

class visualize2D_anim():
    def __init__(self, dir=None,label=0):
        self.dir= dir
        self.label = label
        self.data3D= self.readData2()
        self.predict_name=self.cvtName()

    
    def readData(self):
        lines = open(self.dir,'rt').read().strip().split('\n')
        skeleton_data=[]
        for l in lines:
            frame_data = np.array([float(v) for v in l.strip().split(' ')])
            frame_data = frame_data[1:]
            frame_data = np.reshape(frame_data,(21,3))
            skeleton_data.append(frame_data)
        skeleton_data=np.array(skeleton_data)
    
        #print(skeleton_data.shape)
        return skeleton_data

    def readData2(self):
        lines = open("/media/data3/duc/F-PHAB/FPHAB/fphab_data/"+self.dir,'rt').read().strip().split('\n')
        frame_num = int(len(lines) / 21)
        frameDatas=[]
        fr=[]
        for i,l in enumerate(lines):               
            frame_data = np.array([float(v) for v in l.strip().split(' ')])
            fr.append(frame_data)
        
        for frame in range(frame_num):
                frameData = fr[(frame*21):(frame+1)*21]
                frameData = np.array(frameData)
                frameDatas.append(frameData)
        frameDatas=np.array(frameDatas)
        print('shape:{}'.format(frameDatas.shape))
        return frameDatas

    def realName(self):
        name=self.dir.split('/')[-1]
        name=name.split('_')[0]
        return int(name)
    def anim_skel(self):

        seq = self.data3D
        fig = plt.figure(figsize=[10,10]) 
        ax = fig.add_subplot(111)
        lines = []
        sct=[]
        print(seq.shape)
        N = len(seq)
        data = np.array(list(range(0,N))).transpose()
        #joints order
        joints_order_org=[v-1 for v in [1,2,2,7,7,8,8,9,1,3,3,10,10,11,11,12,1,4,4,13,13,14,14,15,1,5,5,16,16,17,17,18,1,6,6,19,19,20,20,21]]
        joints_order = joints_order_org[::-1]
        #print(len(joints_order))
        skel = seq [0,:,:] 
        #color list
        c=['purple','blue','green','yellow','red']
        c.reverse()
        count_color,count= 0,1

        for id1,id2 in zip(joints_order[::2],joints_order[1::2]):    
            xs, ys = [],[]
            xs=[skel[id1,0],skel[id2,0]]
            ys=[100-skel[id1,1],100-skel[id2,1]]
            line,= plt.plot(xs,ys,color=c[count_color],lw=5)
            scatter=plt.scatter(xs,ys,color=c[count_color],lw=3)
            if(count%4==0):
                count_color+=1
            count+=1
            lines.append(line)
            sct.append(scatter)
        minx,miny=min(seq[0,:,0]),min(seq[0,:,1])
        maxx,maxy=max(seq[0,:,0])-minx,max(seq[0,:,1])-miny
        rect = patches.Rectangle((minx,miny),maxx, maxy, linewidth=1, edgecolor='green', facecolor='none',label="change")
    
        ax.add_patch(rect)
        text=ax.text(minx, maxy, self.predict_name)
        if int(self.realName()-1)==int(self.label):
            bbox_color='green'
        else:
            bbox_color='red'
        print("label:"+str(self.realName()-1)+' '+str(self.label))
        text.set_bbox(dict(facecolor=bbox_color, alpha=0.4))
        plt.grid(False)
        print(seq[0,0,0],seq[0,0,1])
        plt.xlim(seq[0,0,0]-200,seq[0,0,0]+200)
        plt.ylim(100-seq[0,0,1]-200,100-seq[0,0,1]+200)
        plt.title(self.cvt_r_Name(self.realName()))

        #plt.legend(rect,"hii")
        def update(num,data, lines,sct,rect,text):
            for i,line in enumerate(lines):
                segment = np.zeros((2,2))
                joint_1 = joints_order[i*2]
                joint_2 = joints_order[i*2+1]
                #print(joint_1,joint_2)
                xs=[seq[num,joint_1,0],seq[num,joint_2,0]]
               
                ys=[100-seq[num,joint_1,1],100-seq[num,joint_2,1]]
                #print(xs,ys)
                data=np.hstack((xs,ys))
                data=data.reshape(2,2).transpose()
                #print(data)
                line.set_xdata(xs)
                line.set_ydata(ys)
                sct[i].set_offsets(data)
                
            minx,miny=min(seq[num,:,0])-5,min(100-seq[num,:,1])-5
            maxx,maxy=max(seq[num,:,0])-minx+5,max(100-seq[num,:,1])-miny+5
            rect.set_width(maxx)
            rect.set_height(maxy)
            rect.set_xy((minx,miny))
            text.set_position((rect.get_x(),rect.get_height()+3+rect.get_y()))

            #return lines, rect, sct
        anim = animation.FuncAnimation(fig, update, frames=N,fargs=(data,lines,sct,rect,text),interval=100,) 
        nametxt=self.dir.split('/')[-1]
        act,_,s,th=nametxt.split('_')           
        anim.save('/media/data3/duc/F-PHAB/FPHAB/results/vs/'+self.cvtName()+'____s'+s+'_'+th+'.gif',  writer='pillow')
        print('done')


    def cvtName(self):
        label= self.label
        dir= "/media/data3/duc/F-PHAB/FPHAB/label_fphab.txt"
        file_names=open(dir, 'rt').read().strip().split('\n')
        return file_names[int(label)]
    def cvt_r_Name(self,name=0):
        label= name
        dir= "/media/data3/duc/F-PHAB/FPHAB/label_fphab.txt"
        file_names=open(dir, 'rt').read().strip().split('\n')
        return file_names[int(label)-1]