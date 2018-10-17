import cv2
import torch
import numpy as np
import random

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tensorboard import TensorBoard
from model import FCN
from synth import Generator

from bezier import *
from vggnet import *
Encoder = VGG(16, 36)

writer = TensorBoard('log/')
import torch.optim as optim
criterion = nn.MSELoss()
criterion2 = nn.CrossEntropyLoss()

Decoder = FCN(64)
optimizerE = optim.Adam(Encoder.parameters(), lr=3e-4)
optimizerD = optim.Adam(Decoder.parameters(), lr=3e-4)
batch_size = 64
data_size = 100000
generated_size = 0
val_data_size = 512
first_generate = True

use_cuda = True
step = 0
Train_batch = [None] * data_size
Ground_truth = [None] * data_size
Label_batch = [None] * data_size
Val_train_batch = [None] * val_data_size
Val_ground_truth = [None] * val_data_size
Val_label_batch = [None] * val_data_size
G = Generator()

def hisEqulColor(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output

def save_model():
    if use_cuda:
        Decoder.cpu()
        Encoder.cpu()
    torch.save(Decoder.state_dict(),'./Decoder.pkl')
    torch.save(Encoder.state_dict(),'./Encoder.pkl')
    if use_cuda:
        Decoder.cuda()
        Encoder.cuda()

def load_weights():
    Decoder.load_state_dict(torch.load('./Decoder.pkl'))
    # Encoder.load_state_dict(torch.load('./Encoder.pkl'))

def decode(x, train_bezier=False): # b * 36
    x = x.reshape(-1, 9)
    if train_bezier:
        y = Decoder(x.detach())
    else:
        y = None
    x = Decoder(x)
    x = x.reshape(-1, 4, 64, 64)
    return torch.min(x.permute(0, 2, 3, 1), dim=3)[0], y

def sample(n, test=False):
    input_batch = []
    ground_truth = []
    label_batch = []
    if not test:
        batch = random.sample(range(min(data_size, generated_size)), n)
    for i in range(n):
        if test:
            input_batch.append(Val_train_batch[i])
            ground_truth.append(Val_ground_truth[i])
            label_batch.append(Val_label_batch[i])
        else:
            input_batch.append(Train_batch[batch[i]])
            ground_truth.append(Ground_truth[batch[i]])
            label_batch.append(Label_batch[batch[i]])
    input_batch = torch.tensor(input_batch).float()
    ground_truth = torch.tensor(ground_truth).float()
    label_batch =  torch.tensor(np.array(label_batch))
    return input_batch, ground_truth, label_batch

def generate_data():
    print('Generating data')
    global Train_batch, Ground_truth
    global first_generate
    if first_generate == True:
        first_generate = False
        import scipy.io as sio
        mat = sio.loadmat('../data/svhn/train_32x32.mat')
        Data = mat['X']
        Label = mat['y']
        for i in range(len(Label)):
            if Label[i][0] % 10 == 0:
                Label[i][0] = 0
        for i in range(val_data_size):
            img = np.array(Data[..., i])
            origin = noised = img
            origin = cv2.cvtColor(origin, cv2.COLOR_RGB2GRAY)
            origin = cv2.resize(origin, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
            noised = cv2.resize(noised, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
            noised = hisEqulColor(noised) / 255.
            Val_train_batch[i] = noised
            Val_ground_truth[i] = 1 - origin / 255.
            Val_label_batch[i] = Label[i][0]
    global generated_size
    for i in range(1000):
        id = generated_size % data_size
        img, origin, label = G.generate()
        origin = 255. - cv2.cvtColor(origin, cv2.COLOR_RGB2GRAY)
        Train_batch[id] = hisEqulColor(img) / 255.
        Ground_truth[id] = origin / 255.
        Label_batch[id] = label
        generated_size += 1
        
if use_cuda:
    Decoder = Decoder.cuda()
    Encoder = Encoder.cuda()

def train_bezier(x, img):
    Decoder.train()
    x = x.reshape(-1, 9)
    bezier = []
    for i in range(x.shape[0]):        
        bezier.append(draw(x[i]))
    bezier = torch.tensor(bezier).float()
    if use_cuda:
        bezier = bezier.cuda()
    optimizerD.zero_grad()
    loss = criterion(img, bezier)
    loss.backward()
    optimizerD.step()
    Decoder.eval()
    writer.add_scalar('train/bezier_loss', loss.item(), step)
    
def train():    
    Encoder.train()
    train_batch, ground_truth, label_batch = sample(batch_size, test=False)
    train_batch = train_batch.permute(0, 3, 1, 2)
    if use_cuda:
        train_batch = train_batch.cuda()
        ground_truth = ground_truth.cuda()
        label_batch = label_batch.cuda()
    infered_stroke, infered_class = Encoder(train_batch)
    # if step % 5 == 0:
    #    img, stroke_img = decode(infered_stroke, True)
    #    train_bezier(infered_stroke, stroke_img)
    # else:
    img, _ = decode(infered_stroke, False)
    optimizerE.zero_grad()
    loss1 = criterion(img, ground_truth)
    loss2 = criterion2(infered_class, label_batch)
    acc = torch.sum(infered_class.max(1)[1] == label_batch.long()).item() / batch_size
    (loss1 + loss2 * 0.01).backward(retain_graph=True)
    optimizerE.step()
    print('train_loss: ', step, loss1.item(), loss2.item())
    writer.add_scalar('train/img_loss', loss1.item(), step)
    writer.add_scalar('train/class_loss', loss2.item(), step)
    writer.add_scalar('train/acc', acc, step)
    if step % 50 == 0:
        for i in range(10):
            train_img = train_batch[i].cpu().data.numpy()
            gen_img = img[i].cpu().data.numpy()
            ground_truth_img = ground_truth[i].cpu().data.numpy()
            writer.add_image('train/' + str(i) + '/input.png', train_img, step)
            writer.add_image('train/' + str(i) +'/gen.png', gen_img, step)
            writer.add_image('train/' + str(i) +'/ground_truth.png', ground_truth_img, step)

def test():
    Encoder.eval()
    train_batch, ground_truth, label_batch = sample(512, test=True)
    train_batch = train_batch.permute(0, 3, 1, 2)
    if use_cuda:
        train_batch = train_batch.cuda()
        ground_truth = ground_truth.cuda()
        label_batch = label_batch.cuda()
    infered_stroke, infered_class = Encoder(train_batch)
    img, stroke_img = decode(infered_stroke)
    loss = criterion(img, ground_truth)
    print('validate_loss: ', step, loss.item())
    acc = torch.sum(infered_class.max(1)[1] == label_batch.long()).item() / 512
    writer.add_scalar('validate/loss', loss.item(), step)
    writer.add_scalar('validate/acc', acc, step)
    for i in range(10):
        train_img = train_batch[i].cpu().data.numpy()
        gen_img = img[i].cpu().data.numpy()
        ground_truth_img = ground_truth[i].cpu().data.numpy()
        writer.add_image('validate/' + str(i) + '/input.png', train_img, step)
        writer.add_image('validate/' +str(i) +'/gen.png', gen_img, step)
        writer.add_image('validate/' +str(i) +'/ground_truth.png', ground_truth_img, step)

load_weights()                    
while True:
    if step % 100 == 0:
        generate_data()
    train()
    if step % 500 == 0:
        test()
    if step % 1000 == 0:
        save_model()
    step += 1
