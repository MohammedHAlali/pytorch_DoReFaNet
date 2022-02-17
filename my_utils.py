'''convert dataset to grayscale'''

import os
import h5py
import argparse
import numpy as np
from PIL import Image
import torchvision
import torchvision.transforms as transforms

def get_grayscale(sample):
    sample = Image.fromarray(sample)
    new_sample = transforms.Grayscale(num_output_channels=3)(sample)
    print('new sample converted to gs, type(', type(new_sample), ') shape=', new_sample.shape)
    new_sample = transforms.ToTensor()(new_sample)
    print('new sample type: ', type(new_sample))
    print('shape: ', new_sample.shape)
    #new_sample = np.array(new_sample)
    new_sample = new_sample.numpy()
    new_sample = np.swapaxes(new_sample, 0, 2)
    #print('new sample shape: ', new_sample.shape)
    new_sample = new_sample * 255
    new_sample = new_sample.astype('uint8')
    img = Image.fromarray(new_sample)
    return np.array(img)

def get_grayscale_sparsity(sample, s_threshold):
    white_count = 0
    new_sample = transforms.Compose([transforms.ToPILImage()
                ,transforms.Grayscale(num_output_channels=3)
                ,transforms.ToTensor()
                ])(sample)

    img = np.array(new_sample)
    img = np.swapaxes(img, 0, 2)
    img = (img*255).astype('uint8')
    new_img = np.zeros(shape=(img.shape), dtype='uint8')
    for j in range(img.shape[0]):
        for k in range(img.shape[1]):
            #check if RGB pixel color is almost white
            if(img[j,k, 0] >= s_threshold and img[j,k, 1] >= s_threshold and img[j,k, 2] >= s_threshold):
                white_count += 1
                #print('white')
            else:
                new_img[j, k] = img[j, k]
    return new_img

def get_sparsity(sample, s_threshold):
    img = np.array(sample)
    #create blank image of black color
    new_img = np.zeros(shape=(img.shape), dtype='uint8')
    white_count = 0
    for j in range(img.shape[0]):
        for k in range(img.shape[1]):
            #check if RGB pixel color is almost white
            if(img[j,k, 0] >= s_threshold and img[j,k, 1] >= s_threshold and img[j,k, 2] >= s_threshold):
                white_count += 1
            else:
                new_img[j, k] = img[j, k]
    white_ratio = white_count/(img.shape[0]*img.shape[1])
    #apply rotation randomly
    new_img = np.rot90(new_img)
    return new_img, white_ratio

if(__name__ == "__main__"):
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
    if(cfg.color_mode == 'sparsity200'):
        s_threshold = 200
    elif(cfg.color_mode == 'sparsity220'):
        s_threshold = 220
    elif(cfg.color_mode == 'sparsity180'):
        s_threshold = 180
    else:
        s_threshold = 200
    print('sparsity threshold: ', s_threshold)

    white_ratio_list = []
    full_white_count = 0
    for i in range(X.shape[0]):
        sample = X[i]
        label = y[i]
        if(i % 200 == 0):
            print('[{}/{}]'.format(i, X.shape[0]))
            print('sample shape: ', sample.shape, ' type: ', sample.dtype)
            print('lable shape: ', label.shape, ' label = ', label)
        #sample = Image.fromarray(sample)
        #print('image size: ', sample.size, ' type: ', type(sample))
        if(cfg.color_mode == 'rgb'):
            new_X.append(np.array(sample))
            new_y.append(np.array(label))
        elif(cfg.color_mode == 'grayscale'):
            img = get_grayscale(sample)
            if(i % 200 == 0):
                print('new sample shape={}, type={}'.format(img.shape, type(img)))
            new_X.append(np.array(img))
            new_y.append(label)
        elif(cfg.color_mode == 'grayscale_sparsity'):
            img = get_grayscale_sparsity(sample, s_threshold)
            if(i % 200 == 0):
                print('grayscale sparsity image shape={}, avg={}, max={}'.format(img.shape, img.mean(), img.max()))
        
            new_X.append(img)
            new_y.append(label)
        elif('sparsity' in cfg.color_mode):
            img, white_ratio = get_sparsity(sample, s_threshold)
            if(i % 200 == 0):
                print('sparsity image shape={}, avg={}, max={}'.format(img.shape, img.mean(), img.max()))
            new_X.append(img)
            new_y.append(label)
            white_ratio_list.append(white_ratio)
            if(white_ratio >= 0.95):
                print('full white image, white_ratio=', white_ratio)
                full_white_count += 1
                if(label == 1):
                    print('ERROR: label should be 0, but found = ', label)
                    img = Image.fromarray(img)
                    orig_img = Image.fromarray(sample)
                    orig_img.save(os.path.join(path, '{}_{}_rgb_white{}_label_{}.png'.format(
                        cfg.phase, cfg.color_mode, round(white_ratio,3), label)))
                    img.save(os.path.join(path, '{}_{}_sp_white{}_label_{}.png'.format(
                        cfg.phase, cfg.color_mode, round(white_ratio,3), label)))
    
    if('sparsity' in cfg.color_mode):
        print('Number of full (> 95%) white images = ', full_white_count)
        print('length of white ratios: ', len(white_ratio_list))
        print('white ratio average of this dataset: ', np.mean(white_ratio_list))

    print('new X len: ', len(new_X), ' new y len: ', len(new_y))
    X = np.array(new_X)
    y = np.array(new_y)
    print('new data shapes: ', X.shape, y.shape)
    save_name_x = os.path.join(path, '{}_{}_x'.format(cfg.phase, cfg.color_mode))
    save_name_y = os.path.join(path, '{}_{}_y'.format(cfg.phase, cfg.color_mode))
    print('trying to save np: ', save_name_x, save_name_y)
    np.save(save_name_x, X)
    np.save(save_name_y, y)
    print('saved')
