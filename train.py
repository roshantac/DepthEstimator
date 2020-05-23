import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from kornia.losses import SSIM
from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

def train_net(net, device,train_loader, epochs , lr ):
    import datetime
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.01)
    #criterion = nn.BCEWithLogitsLoss()
    criterion = SSIM(3, reduction = 'mean')
    n_train =len(train_loader)
    init = datetime.datetime.now()
    for epoch in range(epochs):
        start =  datetime.datetime.now()
        net.train()
        epoch_loss = 0  
        pbar = tqdm(train_loader)
        #with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:

        for i,batch in enumerate(pbar):
            batch['bg'] = batch['bg'].to(device, dtype = torch.float32)
            batch['fgbg'] = batch['fgbg'].to(device, dtype = torch.float32)
            batch['mask'] = batch['mask'].to(device, dtype = torch.float32)
            batch['depth'] = batch['depth'].to(device, dtype = torch.float32) 
            #depth_pred, masks_pred
            output = net(batch)
            loss1 = criterion(output[1], batch['mask'])
            loss2 = criterion(output[0], batch['depth']) 
            loss = 2*loss1 + loss2 
            epoch_loss += loss.item()
            pbar.set_postfix(desc  = f'Epoch : {epoch+1}  Loss : {loss.item()}  l1: {loss1.item()} l2 = {loss2.item()}')
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()
        scheduler.step() #loss
        end=  datetime.datetime.now()
        print("Time taken for epoch is: ", end-start)
        print(" Total time taken is : ", end -init)
        print("ground truth")
        show(batch['depth'].detach().cpu(),nrow=8)
        print("Depth")
        show(output[0].detach().cpu(),nrow=8) # depth
        print("mask")
        show(output[1].detach().cpu(),nrow=8) #mask
