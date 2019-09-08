
from skimage import io
import numpy as np

def preprocessing(img, mask):
	img = img / 255.
	mask = mask.astype(int)
	return img, mask