from torchreid.utils import FeatureExtractor
import numpy as np
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='log/osnet/osnet_x1_0_imagenet.pth',
    device='cuda'
)

img1 = []
img2 = []

image_list = [
    img1,img2
]

features = extractor(image_list)
# print(features.shape) # output (5, 512)

def cos_sim(a, b):
    a = np.mat(a)
    b = np.mat(b)
    return float(a * b.T) / (np.linalg.norm(a) * np.linalg.norm(b))
