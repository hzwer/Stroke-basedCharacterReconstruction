import cv2
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from tensorboard import TensorBoard
from model import FCN

from bezier import *
writer = TensorBoard('log/')
import torch.optim as optim
criterion = nn.MSELoss()
Decoder = FCN(64)
optimizer = optim.Adam(Decoder.parameters(), lr=3e-4)
batch_size = 64

use_cuda = True
step = 0

def save_model():
    if use_cuda:
        Decoder.cpu()
    torch.save(Decoder.state_dict(),'./Decoder.pkl')
    if use_cuda:
        Decoder.cuda()

def load_weights():
    Decoder.load_state_dict(torch.load('./Decoder.pkl'))

load_weights()
while True:
    Decoder.train()
    train_batch = []
    ground_truth = []
    for i in range(batch_size):
        f = np.random.uniform(0, 1, 9)
        train_batch.append(f)
        ground_truth.append(draw(f))
        
    train_batch = torch.tensor(train_batch).float()
    ground_truth = torch.tensor(ground_truth).float()
    if use_cuda:
        Decoder = Decoder.cuda()
        train_batch = train_batch.cuda()
        ground_truth = ground_truth.cuda()
    gen = Decoder(train_batch)
    optimizer.zero_grad()
    loss = criterion(gen, ground_truth)
    loss.backward()
    optimizer.step()
    print(step, loss.item())
    writer.add_scalar('train/loss', loss.item(), step)    
    if step % 100 == 0:
        Decoder.eval()
        gen = Decoder(train_batch)
        loss = criterion(gen, ground_truth)
        writer.add_scalar('validate/loss', loss.item(), step)
        for i in range(64):
            G = gen[i].cpu().data.numpy()
            GT = ground_truth[i].cpu().data.numpy()
            writer.add_image(str(step) + '/train/gen.png', G, step)
            writer.add_image(str(step) + '/train/ground_truth.png', GT, step)
    if step % 10000 == 0:
        save_model()
    step += 1
