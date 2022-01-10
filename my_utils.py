'''convert dataset to grayscale'''

import os
import h5py
import argparse
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description='Convert PatchCamelyon to grayscale of sparsity')
parser.add_argument('--color_mode', type=str, default='grayscale')
parser.add_argument('--phase', type=str, default='train')
cfg = parser.parse_args()
print('arguments: ', cfg)
path = '../data/camelyon'
base_name = "camelyonpatch_level_2_split_{}_{}.h5"
h5X = h5py.File(os.path.join(path, base_name.format(cfg.phase, 'x')), 'r')
h5y = h5py.File(os.path.join(path, base_name.format(cfg.phase, 'y')), 'r')
X = np.array(h5X.get('x'))
y = np.array(h5y.get('y'))
print('X data shape: ', X.shape, ' type: ', X.dtype)
y = np.squeeze(y)
print('Y data shape: ', y.shape, ' type: ', y.dtype)

new_X = []
new_y = []

for i in range(X.shape[0]):
    print('[{}/{}]'.format(i, X.shape[0]))
    sample = X[i]
    label = y[i]
    print('sample shape: ', sample.shape)
    print('lable shape: ', label.shape, ' label = ', label)
    if(cfg.color_mode == 'grayscale'):
        new_sample = Image.fromarray(sample)
        new_sample = new_sample.convert('L')
        #print('new sample image size: ', new_sample.size)
        new_sample = np.array(new_sample)
        #print('new sample arr shape: ', new_sample.shape)
        new_X.append(new_sample)
        new_y.append(label)
print('new X len: ', len(new_X), ' new y len: ', len(new_y))
X = np.array(new_X)
y = np.array(new_y)
print('new data shapes: ', X.shape, y.shape)
save_name_x = os.path.join(path, '{}_{}_X'.format(cfg.phase, cfg.color_mode))
save_name_y = os.path.join(path, '{}_{}_y'.format(cfg.phase, cfg.color_mode))
print('trying to save np: ', save_name_x, save_name_y)
np.save(save_name_x, X)
np.save(save_name_y, y)
print('saved')

print('done')
