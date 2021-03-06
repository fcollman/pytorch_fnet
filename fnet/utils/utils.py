from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

import collections

import pdb

def delta2rgb(img, extrema = None, cmap = 'PiYG'):
    #converts a difference image (img = im1-im2), into a rgb image where 0 corresponds to black, and positive and negative values are different colors
    
    cmap = plt.get_cmap(cmap)
    cmap_vals = cmap([0,255])
    
    
    cmap_vals = [[1, 0, 1, 1], [0, 1, 1,1]]
    
    
    cmap_vals = np.expand_dims(np.expand_dims(cmap_vals, 2), 3)
    
    img_rgb = np.tile(img, [4, 1, 1])
    
    img_rgb = img_rgb/np.percentile(img,99.9)
    
    
    img_mult_neg = np.tile(cmap_vals[0], [1, img.shape[1], img.shape[2]])
    img_mult_pos = np.tile(cmap_vals[1], [1, img.shape[1], img.shape[2]])
    
    mask_neg = img_rgb<0
    mask_pos = img_rgb>0
    
    
    img_rgb = np.abs(img_rgb)
    
    
    img_rgb[mask_neg] = img_rgb[mask_neg]*img_mult_neg[mask_neg]
    img_rgb[mask_pos] = img_rgb[mask_pos]*img_mult_pos[mask_pos]
    
    #set the alpha layer to "no transparency"
    img_rgb[3,:,:] = 1
    
    img_rgb[img_rgb > 1] = 1
    
    img_rgb = (img_rgb*255).astype('uint8')
    
    return img_rgb
    

def get_stats(pred, target):
    delta = pred - target
    err_map = (delta)**2
    se = np.sum(err_map)
    n_pixels = np.prod(target.shape)
    mse = np.mean(err_map)

    r2 = r2_score(target.flatten(), pred.flatten())
    
    delta_min = np.min(delta) 
    delta_max = np.max(delta)
    
    percentiles = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    percentile_values = np.percentile(np.abs(delta), percentiles)
    
    percentiles = collections.OrderedDict(zip(percentiles, percentile_values))
    
    return err_map, n_pixels, se, mse, r2, delta_min, delta_max, percentiles
        