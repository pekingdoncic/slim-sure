import cv2
import numpy as np

from .dataset import read_image


def draw(row, prediction, output):
    a,b=(read_image(row[key]) for key in ("image0","image1"))
    canvas=np.zeros((max(len(a),len(b)),a.shape[1]+b.shape[1],3),np.uint8)
    canvas[:len(a),:a.shape[1]]=a[:,:,None]
    canvas[:len(b),a.shape[1]:]=b[:,:,None]
    p,q=(np.asarray(prediction[k]).reshape(-1,2) for k in ("xy0","xy1"))
    for i in np.linspace(0,len(p)-1,min(80,len(p)),dtype=int):
        cv2.line(canvas,tuple(np.rint(p[i]).astype(int)),tuple(np.rint(q[i]+[a.shape[1],0]).astype(int)),(70,210,60),1)
    if not cv2.imwrite(str(output),canvas):
        raise OSError(output)
