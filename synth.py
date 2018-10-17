import os
import cv2
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

img_width = 64
img_list = []

for i in range(5000):    
    img = cv2.imread('../digits_resize/digit' + str(i) + '.png')
    img_list.append(img)
    
def rand():
    return np.random.uniform(0, 1)

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
aug_with_origin = iaa.Sequential(
    [
        iaa.OneOf([
            sometimes(iaa.Affine(
            scale={"x": (0.6, 1), "y": (0.6, 1)},
            rotate=(-15, 15), # rotate by -45 to +45 degrees
            shear=(-15, 15), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 0), # if mode is constant, use a cval between 0 and 255
            mode='constant'
            )),
            sometimes(iaa.Pad(
                percent=(0, 0.4),
                pad_mode='constant',
                pad_cval=(0, 0)
            )),
        ]),
        iaa.OneOf([
            sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1))),
            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
            ])
    ]
)
aug = iaa.Sequential(
    [
        iaa.SomeOf((0, 6),
                   [
                       (iaa.Superpixels(p_replace=(0, 1.0), n_segments=(100, 200))),
                       # convert images into their superpixel representation
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                           iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                           iaa.MedianBlur(k=(3, 9)), # blur image using local medians with kernel sizes between 2 and 7
                       ]),
                       iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                       iaa.Grayscale(alpha=(0.0, 1.0)),
                       iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                       iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                       iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                       iaa.Invert(0.5, per_channel=True), # invert color channels
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                       iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                       iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25), # move pixels locally around (with random strengths)                       
                   ],
                 random_order=True
                 ),
    sometimes(iaa.OneOf([
        iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
        iaa.SaltAndPepper(p=(0.03, 0.3)),
    ])),
        ]
    )


def uneven_light():
    def norm(Z):
        min_in_z=np.min(Z)
        max_in_z=np.max(Z)
        Z=(Z-min_in_z)/(max_in_z-min_in_z)
        return Z

    def rotate(x,y,cos):
        sin=np.sqrt(1-cos**2)
        
        x_=cos*x+sin*y
        y_=-sin*x+cos*y
    
        return x_,y_

    def scale(x,y,s1,s2):
        x_,y_=x*s1,y*s2
        return x_,y_

    def translate(x,y,x0,y0):
        return x+x0,y+y0
    
    w,h=img_width,img_width
    
    x=np.linspace(-1,1,w)
    y=np.linspace(-1,1,h)
    
    r=np.random.rand()
    s1,s2=np.random.rand()*5,np.random.rand()*5
    t1,t2=np.random.rand()*5,np.random.rand()*5
    
    X, Y = np.meshgrid(x, y)
    X1,Y1=rotate(X,Y,r)
    X1,Y1=scale(X1,Y1,s1,s2)
    X1,Y1=translate(X1,Y1,t1,t2)
    
    Z=(np.exp(Y1)+np.exp(-Y1))/2*np.sin(X1)
    Z=norm(Z)
    return Z

class Generator:
    def __init__(self):
        pass
        
    def center_square_crop(self, img, wx, wy, cx=0, cy=0, origin=None):
        h = img.shape[0]
        w = img.shape[1]
        if cx:
            x = np.max([cx - wx // 2, 0])
            y = np.max([cy - wy // 2, 0])
        else:
            x = (h - wx) // 2
            y = (w - wy) // 2
        img = img[x : x + wx, y : y + wy]
        if type(img) == type(origin):
            origin = origin[x : x + wx, y : y + wy]
        img = cv2.resize(img, dsize=(h, w), interpolation=cv2.INTER_CUBIC)
        origin = cv2.resize(origin, dsize=(h, w), interpolation=cv2.INTER_CUBIC)
        return img, origin    

    def random_rotate(self, img, ang, origin = None):
        ang = np.random.randint(-ang, ang)
        x = img.shape[0]
        y = img.shape[1]
        center = (y // 2, x // 2)
        M = cv2.getRotationMatrix2D(center, ang, 0.8)
        img = cv2.warpAffine(img, M, (y, x))
        if type(img) == type(origin):
            origin = cv2.warpAffine(origin, M, (y, x))
        return img, origin

    def random_color(self, h, w, same=False):
        R = np.ones((h,w)) * rand()
        G = np.ones((h,w)) * rand()
        B = np.ones((h,w)) * rand()
        L = uneven_light()
        R += L * np.random.uniform(-1, 1)
        G += L * np.random.uniform(-1, 1)
        B += L * np.random.uniform(-1, 1)
        if same:
            return np.stack((R, R, R), axis=2)
        else:
            return np.stack((R, G, B), axis=2)

    def generate_color(self):
        if rand() < 0.2:
            C1 = self.random_color(img_width, img_width, same=True)
        else:
            C1 = self.random_color(img_width, img_width)
        if rand() < 0.2:
            C2 = self.random_color(img_width, img_width, same=True)
        else:
            C2 = self.random_color(img_width, img_width)
        return C1, C2
    
    def add_color(self, img):
        C1, C2 = self.generate_color()
        while np.max(np.abs(C1 - C2) < 0.3):
            C1, C2 = self.generate_color()
        img = C1 * img + C2 * (1 - img)
        img = np.clip(img, 0, 1)
        return img
    
    def forgery_data(self, img1, img2, img3):
        if np.random.rand() < 0.2:
            img1 = np.zeros([64, 48, 3])
        if np.random.rand() < 0.2:
            img3 = np.zeros([64, 48, 3])
        img = np.hstack([np.array(img1), np.array(img2), np.array(img3)]).astype('uint8')
        origin = np.hstack([np.zeros(img1.shape), np.array(img2), np.zeros(img3.shape)]).astype('uint8')
        cy = img1.shape[1] + img2.shape[1] // 2
        cx = img2.shape[0] // 2
        # print(img.shape, cx, cy)
        h = np.random.randint(54, 64 + 1)
        w = np.random.randint(img2.shape[1], min(img2.shape[1] * 5, origin.shape[1] + 1))
        img, origin = self.center_square_crop(img, 64, w, cx, cy, origin)
        img = cv2.resize(img, dsize=(img_width, img_width), interpolation=cv2.INTER_CUBIC)
        origin = cv2.resize(origin, dsize=(img_width, img_width), interpolation=cv2.INTER_CUBIC)
        # img, origin = self.random_rotate(img, 20, origin)
        # print(img.dtype, origin.dtype)
        img = (self.add_color(img / 255.) * 255).astype('uint8')
        tmp = np.concatenate([img, origin], 2)
        tmp = aug_with_origin.augment_image(tmp)
        img = tmp[:, :, 0 : 3]
        origin = tmp[:, :, 3 : 6]
        img = aug.augment_image(img)
        return img, origin
    
    def generate(self):
        label0 = np.random.randint(10)
        label1 = np.random.randint(10)
        label2 = np.random.randint(10)
        img0 = img_list[np.random.randint(500) * 10 + label0]
        img1 = img_list[np.random.randint(500) * 10 + label1]
        img2 = img_list[np.random.randint(500) * 10 + label2]
        label1 += 1
        if label1 == 10:
            label1 = 0
        img, origin = self.forgery_data(img0, img1, img2)
        return img, origin, label1
        
if __name__ == '__main__':
    G = Generator()
    for i in range(100):
        img, origin, label = G.generate()
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        origin = cv2.resize(origin, (224, 224), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('img', img)
        cv2.imshow('ground_truth', origin)
        cv2.waitKey(0)
