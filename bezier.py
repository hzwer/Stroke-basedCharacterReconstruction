import cv2
import numpy as np

canvas_width = 256
output_width = 64

def normal(x):
    return (int)(x * (canvas_width - 1) + 0.5)

def draw(f):
    x0, y0, z0, x1, y1, z1, x2, y2, z2 = f
    x0 = normal(x0)
    x1 = normal(x1)
    x2 = normal(x2)
    y0 = normal(y0)
    y1 = normal(y1)
    y2 = normal(y2)
    z0 = (int)(z0 * 32 + 2)
    z1 = (int)(z1 * 32 + 2)
    z2 = (int)(z2 * 32 + 2)
    canvas = np.zeros([canvas_width, canvas_width]).astype('float32')
    tmp = 1. / 100
    for i in range(100):
        t = i * tmp
        x = (int)((1-t) * (1-t) * x0 + 2 * t * (1-t) * x1 + t * t * x2)
        y = (int)((1-t) * (1-t) * y0 + 2 * t * (1-t) * y1 + t * t * y2)
        z = (int)((1-t) * (1-t) * z0 + 2 * t * (1-t) * z1 + t * t * z2)
        cv2.circle(canvas, (y, x), z, 1., -1)
    return 1 - cv2.resize(canvas, dsize=(output_width, output_width))
