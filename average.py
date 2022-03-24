from model import EventDetector
from dataloader_T import GolfDB_T, Normalize_T
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from two_stream_dataloader import GolfDB_2_stream
from dataloader13 import GolfDB_13, ToTensor_13, Normalize_13, Normalize_T_13
import torch.nn.functional as F
import numpy as np
from util import correct_preds
import matplotlib.pyplot as plt
from data.config import cfg

# summary = [0, 0, 0, 0, 0, 0, 0, 0]  #统计各个关键帧检测出错的数目


def myeval(model_rgb, model_opt, split, seq_length, n_cpu, disp, stream_choice=0):
    # summaryFile = open("summary_opt_{}.txt".format(split),"w")
    videosNum = 0  # 统计验证集的视频数量
    dataset = GolfDB_2_stream(data_file='./data/val_split_{}.pkl'.format(split),
                              vid_dir=cfg.OPT_RESIZE_FILE_PATH,
                              seq_length=seq_length,
                              train=False)
    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=n_cpu,
                             drop_last=False)

    correct = []
    # 这三个是为了做融合用
    all_probs = []
    all_tols = []
    all_events = []

    for i, sample in enumerate(data_loader):
        videosNum += 1
        images, opt_images, labels = sample['images'], sample['opt_images'], sample['labels']
        # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
                opt_image_batch = opt_images[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch *
                                     seq_length:(batch + 1) * seq_length, :, :, :]
                opt_image_batch = opt_images[:, batch *
                                     seq_length:(batch + 1) * seq_length, :, :, :]
                                     
            logits_rgb = model_rgb(image_batch.cuda())
            logits_opt = model_opt(opt_image_batch.cuda())
            if batch == 0:
                probs_rgb = F.softmax(logits_rgb.data, dim=1).cpu().numpy()
                probs_opt = F.softmax(logits_opt.data, dim=1).cpu().numpy()
            else:
                probs_rgb = np.append(probs_rgb, F.softmax(
                    logits_rgb.data, dim=1).cpu().numpy(), 0)
                probs_opt = np.append(probs_opt, F.softmax(
                    logits_opt.data, dim=1).cpu().numpy(), 0)
            batch += 1
        probs_opt = np.multiply(probs_opt,1.18)
        probs = np.add(probs_rgb,probs_opt)
        probs = np.divide(probs,2.0)
        events, preds, _, tol, c = correct_preds(probs, labels.squeeze())
        
        all_probs.append(probs)
        all_tols.append(tol)
        all_events.append(events)
        
        if disp:
            # print(i, c)
            print("ground truth:")
            print(events)
            print("preds:")
            print(preds)
        correct.append(c)
    PFCR = np.mean(correct,axis=0)
    PCE = np.mean(correct)
    print("PCE:")
    print(PCE)
    print("PFCR")
    print(PFCR)
    # summaryFile.close()
    return PCE, videosNum, all_probs, all_tols, all_events,PFCR


if __name__ == '__main__':

    split = cfg.SPLIT
    seq_length = cfg.SEQUENCE_LENGTH
    n_cpu = cfg.CPU_NUM

    model_rgb = EventDetector(pretrain=True,
                          width_mult=1,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)
    model_opt = EventDetector(pretrain=True,
                          width_mult=1,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)
    
    PCES = {}
    PFCRS = {}
    vNum = 0
    
    # rgb
    save_dict = torch.load('./final_model/swingnet_{}.pth.tar'.format(1500))
    new_state_dict = save_dict['model_state_dict']
    model_rgb.load_state_dict(new_state_dict)
    model_rgb.cuda()
    model_rgb.eval()
    # opt
    save_dict = torch.load('./rgb_opt_stream/swingnet_{}.pth.tar'.format(1700))
    new_state_dict = save_dict['model_state_dict']
    model_opt.load_state_dict(new_state_dict)
    model_opt.cuda()
    model_opt.eval()
    
    PCE, vNum, _, _, _ ,PFCR= myeval(
            model_rgb, model_opt, split, seq_length, n_cpu, False, 0)
        
    

    # nothing
