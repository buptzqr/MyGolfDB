from model import EventDetector
from dataloader_T import GolfDB_T, Normalize_T
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import GolfDB, ToTensor, Normalize
import torch.nn.functional as F
import numpy as np
from util import correct_preds
import matplotlib.pyplot as plt 


def eval(model, split, seq_length, n_cpu, disp):
    #评价非光流法
    # dataset = GolfDB(data_file='data/val_split_{}.pkl'.format(split),
    #                  vid_dir='data/videos_160/',
    #                  seq_length=seq_length,
    #                  transform=None,
    #                  myMean=[0.485, 0.456, 0.406],
    #                  myStd=[0.229, 0.224, 0.225],
    #                  train=False)

    #评价光流法
    dataset = GolfDB_T(data_file='data/train_split_{}.pkl'.format(split),
                        vid_dir='/home/zqr/codes/data/opticalFlowRes_160',
                        seq_length=seq_length,
                        train=False)
    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=n_cpu,
                             drop_last=False)

    correct = []

    for i, sample in enumerate(data_loader):
        images, labels = sample['images'], sample['labels']
        # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
            logits = model(image_batch.cuda())
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1
        _, _, _, _, c = correct_preds(probs, labels.squeeze())
        if disp:
            print(i, c)
        correct.append(c)
    PCE = np.mean(correct)
    return PCE


if __name__ == '__main__':

    split = 4
    seq_length = 64
    n_cpu = 6

    model = EventDetector(pretrain=True,
                          width_mult=1,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)
    PCES={}
    for i in range(10,26):
        index=i*100
        print('models/swingnet_{}.pth.tar'.format(index))    
        save_dict = torch.load('models/swingnet_{}.pth.tar'.format(index))
        model.load_state_dict(save_dict['model_state_dict'])
        model.cuda()
        model.eval()
        PCE = eval(model, split, seq_length, n_cpu, False)
        PCES[index] = PCE

    print('split:{}  Average PCE: {}'.format(split,PCES))
    
    y_val = list(PCES.values())
    x_val = list(PCES.keys()) 

    plt.plot(x_val, y_val, linewidth=5) 

    #设置图表标题，并给坐标轴加上标签 
    plt.title("val_precision", fontsize=24) 
    plt.xlabel("iter per 100", fontsize=14)
    plt.ylabel("acc val", fontsize=14) 

    #设置刻度标记的大小 
    plt.tick_params(axis='both', labelsize=14) 
    plt.savefig("split{}".format(split))

