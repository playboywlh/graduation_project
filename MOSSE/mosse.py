#MOSSE (refer to "Visual Object Tracking using Adaptive Correlation Filters")

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

class mosse:
    def __init__(self,image_path,args=None):
        self.args={'sigma':2,'lr':0.125}
        if args:
            self.args.update(args)
        self.image_path=image_path
        self.frame_list=self._get_frame_list()
        self.frame_list.sort()
        self.sigma=self.args['sigma']
        self.lr=self.args['lr']
        self.num_pretrain=20
        self.num_training=1 

    def mosse_init(self,init_frame,init_gt):
        response_map=self._get_gauss_response(init_frame,init_gt)
        gi=response_map[init_gt[1]:init_gt[1]+init_gt[3],init_gt[0]:init_gt[0]+init_gt[2]]
        init_frame_clip=init_frame[init_gt[1]:init_gt[1]+init_gt[3],init_gt[0]:init_gt[0]+init_gt[2]]
        self.window=self._window_func(init_gt)
        fi=self._pre_process(init_frame_clip)
        self.Fi=np.fft.fft2(fi)
        print('fi',fi)
        self.Gi=np.fft.fft2(gi)
        self.Ai=self.Gi*np.conjugate(self.Fi)
        self.Bi=self.Fi*np.conjugate(self.Fi)
        self._pre_training(init_frame_clip)

    def _pre_training(self,frame):
        #height,width=self.Gi.shape
        #frame=cv2.resize(frame,(width,height))
        for _ in range(self.num_pretrain-1):
            fi=self._pre_process(self._random_warp(frame))
            print(fi)
            self.Fi=np.fft.fft2(fi)
            self.Ai=self.Ai+self.Gi*np.conjugate(self.Fi)
            self.Bi=self.Bi+self.Fi*np.conjugate(self.Fi)
        self.Ai/=self.num_pretrain
        self.Bi/=self.num_pretrain
    def _random_warp(self,image):
        a=-180/6
        b=180/6
        r=a+(b-a)*np.random.uniform()

        matrix_rot=cv2.getRotationMatrix2D((image.shape[1]/2,image.shape[0]/2),r,1)
        image_rot=cv2.warpAffine(image,matrix_rot,(image.shape[1],image.shape[0]))
        cv2.imshow('image_rot',image_rot.astype(np.uint8))
        cv2.waitKey()
        return image_rot

    def _pre_process(self,frame):
        if len(frame.shape)==3:
            image=frame.swapaxes(0,2).swapaxes(1,2).astype(np.float32)
        else:
            image=frame.astype(np.float32)
        height,width=image.shape[-2:]
        image=np.log(image+1)
        print('image',image)
        print('image_mean',np.mean(image))
        print('image_std',np.std(image))
        image=(image-np.mean(image))/(np.std(image)+1e-5)
        print(image.shape)
        window=self.window
        image=image*window
        image=self._linear_mapping(image)
        print('win_img',image)
        return image

    def _window_func(self,gt):
        height,width=gt[-2:]
        col=np.hanning(height)
        row=np.hanning(width)
        mask_col,mask_row=np.meshgrid(col,row)
        window=mask_col*mask_row
        return window
    
    def mosse_update(self,frame):
        Ai=self.Gi*np.conjugate(self.Fi)
        Bi=self.Fi*np.conjugate(self.Fi)
        for _ in range(self.num_training-1):
            fi=self._pre_process(self._random_warp(frame))
            Fi=np.fft.fft2(fi)
            Ai=Ai+self.Gi*np.conjugate(Fi)
            Bi=Bi+Fi*np.conjugate(Fi)
        Ai/=self.num_training
        Bi/=self.num_training
        self.Ai=self.lr*Ai+(1-self.lr)*self.Ai
        self.Bi=self.lr*Bi+(1-self.lr)*self.Bi
        #self.Ai=self.lr*self.Gi*np.conjugate(self.Fi)+(1-self.lr)*self.Ai
        #self.Bi=self.lr*self.Fi*np.conjugate(self.Fi)+(1-self.lr)*self.Bi

    def mosse_track(self):
        init_frame=cv2.imread(self.frame_list[0])
        #init_frame=cv2.cvtColor(init_frame,cv2.COLOR_BGR2GRAY)
        init_gt=cv2.selectROI('demo',init_frame,False,False)
        init_gt=np.array(init_gt).astype(np.int32)
        self.mosse_init(init_frame,init_gt)
        pos=init_gt.copy()
        clip_pos=np.array([init_gt[0],init_gt[1],init_gt[0]+init_gt[2],init_gt[1]+init_gt[3]])

        for i in range(1,len(self.frame_list)):
            current_frame=cv2.imread(self.frame_list[i])
            #current_frame=cv2.cvtColor(current_frame,cv2.COLOR_BGR2GRAY)
            current_clip=current_frame[clip_pos[1]:clip_pos[3],clip_pos[0]:clip_pos[2]].astype(np.float32)
            current_frame_clip=cv2.resize(current_clip,(init_gt[2],init_gt[3]))
            fi=self._pre_process(current_frame_clip)
            self.Fi=np.fft.fft2(fi)
            self.Hi=self.Ai/self.Bi
            self.Gi=self.Hi*self.Fi
            gi=np.fft.ifft2(self.Gi)
            gi=self._linear_mapping(gi)
            max_value=gi.max(axis=1).max(axis=1)
            max_pos=np.zeros([3,2])
            max_pos[0]=np.where(gi[0]==max_value[0])
            max_pos[1]=np.where(gi[1]==max_value[1])
            max_pos[2]=np.where(gi[2]==max_value[2])
            mean_pos=np.mean(max_pos,axis=0)
            dy=int(np.mean(mean_pos[0]-gi.shape[1]/2))
            dx=int(np.mean(mean_pos[1]-gi.shape[2]/2))
            pos[0]=pos[0]+dx
            pos[1]=pos[1]+dy
        
            clip_pos[0]=np.clip(pos[0],0,current_frame.shape[1])
            clip_pos[1]=np.clip(pos[1],0,current_frame.shape[0]) 
            clip_pos[2]=np.clip(pos[0]+pos[2],0,current_frame.shape[1]) 
            clip_pos[3]=np.clip(pos[1]+pos[3],0,current_frame.shape[0]) 

            self.mosse_update(current_frame_clip)

            cv2.rectangle(current_frame,(pos[0],pos[1]),(pos[0]+pos[2],pos[1]+pos[3]),(255,0,0),2)
            cv2.imshow('demo',current_frame.astype(np.uint8))
            k=cv2.waitKey()
            if k==27:
                cv2.destroyAllWindows()
                exit()

    def _get_gauss_response(self,image,gt):
        height,width=image.shape[:2]
        xx,yy=np.meshgrid(np.arange(width),np.arange(height))
        center_x=gt[0]+0.5*gt[2]
        center_y=gt[1]+0.5*gt[3]
        dist=(np.square(xx-center_x)+np.square(yy-center_y))/(np.square(self.sigma))
        response=np.exp(-dist)
        response=self._linear_mapping(response)
        return response

    def _linear_mapping(self,image):
        return (image-image.min())/(image.max()-image.min())

    def _get_frame_list(self):
        frame_list=[]
        for item in os.listdir(self.image_path):
            if item.endswith('.jpg'):
                path=os.path.join(self.image_path,item)
                frame_list.append(path)
        return frame_list

if __name__=='__main__':
    tracker=mosse('/media/pci/4T/wlh/Siammask/data/VOT2016/hand')
    tracker.mosse_track()
