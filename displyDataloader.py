import os.path as osp
import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

#非光流部分
def ToTensor(sample):
    """Convert ndarrays in sample to Tensors."""
    images, labels = sample['images'], sample['labels']
    # images= np.asarray(images)
    nImages=[]
    for image in images:
        if isinstance(image,np.ndarray):
            pass
        else:
            image=image.numpy()
        nImages.append(image)
        # nImages.append(image)
    images= np.asarray(nImages)    
    images = images.transpose((0, 3, 1, 2))
    return {'images': torch.from_numpy(images).float().div(255.),
            'labels': torch.from_numpy(labels).long()}


def Normalize(sample,mean,std):
    images, labels = sample['images'], sample['labels']
    images.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return {'images': images, 'labels': labels}


class GolfDB(Dataset):
    def __init__(self, data_file, vid_dir, seq_length, train=True,transform=None, myMean=[], myStd=[]):
        self.dataInfo = open(data_file)
        self.vid_dir = vid_dir
        self.seq_length = seq_length
        self.transform = transform
        self.myMean=myMean
        self.myStd=myStd
        self.videosPath=[]
        for line in self.dataInfo:
            line = line.rstrip()
            info = line.split()
            self.videosPath.append(info[0])


    def __len__(self):
        return len(self.videosPath)

    def __getitem__(self, idx):
        images = []
        labels = []
        videoPath = self.videosPath[idx]
        cap = cv2.VideoCapture(videoPath)

        
            # full clip
        for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            _, img = cap.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            transImg = np.asarray(img)
            if self.transform:
                transImg=self.transform(transImg)
            images.append(transImg)
            # cv2.imwrite(osp.join("/home/zqr/codes/data/transImg","transImage.jpg"), transImg)
            labels.append(8)
        cap.release()
        
        sample = {'images':images, 'labels':np.asarray(labels)}
        # sample = {'images':np.asarray(images), 'labels':np.asarray(labels)}
        # if self.transform:
        #     sample['images'] = self.transform(sample['images'])
        sample=ToTensor(sample)
        # sample=Normalize(sample)
        return sample


#光流部分
def Normalize_T(sample):
    images, labels = sample['images'], sample['labels']
    images = np.asarray(images)
    labels = np.asarray(labels)
    # print(images.shape)
    imgsMean = np.mean(images, axis=(1, 2))
    imgsMean = imgsMean.reshape(-1, 1, 1, 3)
    images = np.subtract(images, imgsMean)
    images = images.transpose((0, 3, 1, 2))
    images = torch.from_numpy(images)
    labels = torch.from_numpy(labels)
    return {'images': images, 'labels': labels}


class GolfDB_T(Dataset):
    def __init__(self, data_file, vid_dir, seq_length, transform=None, train=True):
        self.df = pd.read_pickle(data_file)
        self.vid_dir = vid_dir
        self.seq_length = seq_length
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        a = self.df.loc[idx, :]  # annotation info
        events = a['events']
        events -= events[0]  # now frame #s correspond to frames in preprocessed video clips

        images, labels = [], []
        opticalFileFolder = osp.join(self.vid_dir, '{}'.format(a['id']))
        # print(opticalFileFolder)
        if self.train: 
            start_frame = np.random.randint(events[-1] + 1)
            pos = start_frame
            #光流文件是从第一帧开始的
            if pos == 0:
                pos = 1
            while len(images) < self.seq_length:
                opticalFileName = osp.join(opticalFileFolder, '{:0>4d}.flo'.format(pos))
                if os.path.exists(opticalFileName):
                    opticalOri = np.fromfile(opticalFileName, np.float32, offset=12).reshape(160, 160, 2)
                    opticalArray = np.empty([160, 160, 3], np.float32)
                    opticalArray[..., 0] = 255
                    opticalArray[..., 1] = opticalOri[:,:,0]
                    opticalArray[..., 2] = opticalOri[:,:,1]
                    if self.transform:
                        opticalArray=self.transform(opticalArray)
                    images.append(opticalArray)
                    if pos in events[1:-1]:
                        labels.append(np.where(events[1:-1] == pos)[0][0])
                    else:
                        labels.append(8)
                    pos += 1
                else:
                    pos = 1
        else:
            # full clip
            #get files num
            filesNum = -1
            for ger in os.walk(opticalFileFolder):
                filesNum = len(ger[2])
            for pos in range(1,filesNum + 1):
                opticalFileName = osp.join(opticalFileFolder,'{:0>4d}.flo'.format(pos))
                # print(opticalFileName)
                opticalOri = np.fromfile(opticalFileName, np.float32, offset=12).reshape(160, 160, 2)
                opticalArray = np.empty([160, 160, 3], np.float32)
                opticalArray[..., 0] = 255
                opticalArray[..., 1] = opticalOri[:,:,0]
                opticalArray[..., 2] = opticalOri[:,:,1]
                # print(opticalFileName + "is ok")
                if self.transform:
                    opticalArray=self.transform(opticalArray)
                images.append(opticalArray)
                if pos in events[1:-1]:
                    labels.append(np.where(events[1:-1] == pos)[0][0])
                else:
                    labels.append(8)
        sample = {'images': images, 'labels': np.asarray(labels)}
        sample = Normalize_T(sample)
        
        return sample



if __name__ == '__main__':

    myMean=[0.485, 0.456, 0.406]
    myStd=[0.229, 0.224, 0.225]  # ImageNet mean and std (RGB)
    
    dataset = GolfDB(data_file='/home/zqr/codes/GolfDB/data/videos_processed_160/datafile.txt',
                     vid_dir='/home/zqr/codes/GolfDB/data/videos_processed_160',
                     seq_length=64,
                     transform=transforms.Compose([transforms.ToPILImage(),
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.RandomAffine(5,shear=5),
                        transforms.ToTensor()]),
                     train=False,
                     myMean=myMean,
                     myStd=myStd)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=6, drop_last=False)

    for i, sample in enumerate(data_loader):
        images, labels = sample['images'], sample['labels']
        events = np.where(labels.squeeze() < 8)[0]
        print('{} events: {}'.format(len(events), events))




    





       

