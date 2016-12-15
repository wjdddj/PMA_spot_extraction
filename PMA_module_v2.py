#!/Users/jwang/anaconda/bin/

import matplotlib.pyplot as plt
import numpy as np
import pandas
import os
import re
from time import localtime, strftime
from scipy import stats, ndimage

def grep_tif(files):
    img_files = []
    for file in files:
        if re.search('\\.tif$|\\.tiff$', file):
            img_files.append(file)
    return img_files

def read_images(img_dir):
    # print 'reading images...'
    files = os.listdir(img_dir)
    img_files = grep_tif(files)
    img_dic = {}
    for img_file in img_files:
        print 'reading images...%s' % img_file
        im_name = img_file.split('.')[0]
        img_dic[im_name] = plt.imread(img_dir + img_file)
    print 'finished reading images!\n'
    return img_dic

def add_margin(im_data, size):
    x_len, y_len = im_data.shape
    bg_signal = np.percentile(im_data, 30) # estimate background using 30 percentile of the image
    bg_data = np.array(np.repeat(bg_signal, x_len), dtype = im_data.dtype)
    # insert background signals in front of the image
    for i in range(size):
        im_data = np.insert(im_data, 0, bg_data, 1)
    # append background signals at the end of the image
    for i in range(size):
        im_data = np.insert(im_data, y_len+size, bg_data, 1)
    return im_data

def find_centroid(big_spot):
    x_center, y_center = np.array(big_spot.shape)/2
    x_len, y_len = big_spot.shape
    
    # compute edge by two dimensional first order derivatives
    bigX, bigY = np.gradient(np.log(big_spot))
    idx_left_edge = int(stats.mode(np.argmax(bigY[:,:y_center], axis = 1))[0])
    idx_right_edge = int(stats.mode(np.argmin(bigY[:,y_center:(y_len-10)], axis = 1))[0]) + y_center
    idx_upper_edge = int(stats.mode(np.argmax(bigX[:x_center,:], axis = 0))[0])
    idx_lower_edge = int(stats.mode(np.argmin(bigX[x_center:(x_len-10),:], axis = 0))[0]) + x_center
    
    x_new = (idx_left_edge + idx_right_edge)/2
    y_new = (idx_upper_edge + idx_lower_edge)/2
    centroid_new = np.array([x_new, y_new])
    return centroid_new

def update_centroid(big_spot, centroid):
    x = centroid[0] # definition by grid file
    y = centroid[1] # definition by grid file
    # xmax, ymax = big_spot.shape
    edge_threshold = -0.3
    
    up_gradient = np.gradient(np.log(big_spot[:x,y][::-1]))
    lo_gradient = np.gradient(np.log(big_spot[x:,y]))
    lf_gradient = np.gradient(np.log(big_spot[x,:y][::-1]))
    rt_gradient = np.gradient(np.log(big_spot[x,y:]))
    
    d_upper_edge = len(up_gradient) if(min(up_gradient) > edge_threshold) else np.argmin(up_gradient)
    d_lower_edge = len(lo_gradient) if(min(lo_gradient) > edge_threshold) else np.argmin(lo_gradient)
    d_left_edge = len(lf_gradient) if(min(lf_gradient) > edge_threshold) else np.argmin(lf_gradient)
    d_right_edge = len(rt_gradient) if(min(rt_gradient) > edge_threshold) else np.argmin(rt_gradient)

    x_new = ((2*x - d_left_edge + d_right_edge))/2
    y_new = ((2*y - d_upper_edge + d_lower_edge))/2
    centroid_new = np.array([x_new, y_new])
    return centroid_new

def adjust_grid(im_data, grid):
    n_spot = grid.shape[0]
    radius = 500
    grid_new = pandas.DataFrame(index=range(0, n_spot), columns = ['grid X', 'grid Y'])
    #grid_new['spot_number'] = range(1, n_spot+1)
    for i in range(0,n_spot):
        #print 'adjusting spot %s' %i
        centroid = np.array([grid['grid X'][i], grid['grid Y'][i]])
        big_spot = get_spot(im_data, centroid, radius)
        big_spot = ndimage.filters.gaussian_filter(big_spot, sigma = 5, order = 0) # apply gaussian filter before adjusting grid
        coords_shift = centroid - radius
        coords_shift[0] = max(coords_shift[0], 0)
        coords_shift[1] = max(coords_shift[1], 0)
        #centroid_new = update_centroid(big_spot, (radius, radius))
        centroid_new = find_centroid(big_spot)
        centroid_new = centroid_new + coords_shift
        grid_new['grid X'][i] = centroid_new[0]
        grid_new['grid Y'][i] = centroid_new[1]
    return grid_new

def get_spot(im_data, centroid, radius):
    xmax = im_data.shape[1]
    ymax = im_data.shape[0]
    x1 = max(int(centroid[0] - radius), 0)
    x2 = min(int(centroid[0] + radius), xmax)
    y1 = max(int(centroid[1] - radius), 0)
    y2 = min(int(centroid[1] + radius), ymax)
    spot = im_data[y1:y2, x1:x2]
    return spot

def get_spot_data(im_data, grid):
    #now = strftime("%Y%m%d", localtime())
    n_spot = grid.shape[0]
    spots = {}
    for i in range(0,n_spot):
        spots[i] = get_spot(im_data, (grid['grid X'][i], grid['grid Y'][i]), 250)
    spot_median = {k:np.median(v) for k, v in spots.items()}
    spot_sum = {k:np.sum(v) for k, v in spots.items()}
    out_data = {'spot_number':range(1,65), 'median':spot_median.values(), 'sum':spot_sum.values()}
    out_data = pandas.DataFrame(out_data)
    out_data = out_data[['spot_number', 'median', 'sum']]
    return out_data

def draw_grid_box(centroid, radius):
    x1 = centroid[0] - radius
    x2 = centroid[0] + radius
    y1 = centroid[1] - radius
    y2 = centroid[1] + radius
    draw_box(x1, x2, y1, y2)
    return

def draw_box(x1, x2, y1, y2):
    plt.plot((x1, x2), (y1, y1), color='r', linewidth = 0.2)
    plt.plot((x1, x2), (y2, y2), color='r', linewidth = 0.2)
    plt.plot((x1, x1), (y1, y2), color='r', linewidth = 0.2)
    plt.plot((x2, x2), (y1, y2), color='r', linewidth = 0.2)
    return

def draw_array(im_data, grid):
    plt.imshow(np.log(im_data), cmap = 'Greys')
    plt.axis('off')
    for i in range(0,64):
        centroid = np.array([grid['grid X'][i], grid['grid Y'][i]])
        draw_grid_box(centroid, 250)
    return


def main():
    
    working_dir = os.getcwd() + '/'
    grid_file = 'grid_template.csv'
    
    now = strftime("%Y%m%d", localtime())
    
    # read in initial grid file
    grid = pandas.read_csv(working_dir + grid_file)
    
    # read in images
    im_dic = read_images(working_dir)
    
    # define margin size
    margin_size = 5
    
    # add margin to each image
    print 'adding margins to each image...'
    for im_name, im_data in  im_dic.items():
        im_data = add_margin(im_data, margin_size)
        im_dic[im_name] = im_data
    
    
    # adjusting grid for each slide, alignment
    print ''
    grid_adjust = {}
    for im_name, im_data in im_dic.items():
        print 'adjusting grid for %s' % im_name
        # adjusting grid for each slide
        grid_adjust[im_name] = adjust_grid(im_data, grid)
        grid_adjust[im_name].to_csv(working_dir + now + '_' + im_name + '_grid_out.csv')
    
    # obtain data based on aligned grid
    print ''
    out_sum = out_median = pandas.DataFrame({'spot_number':range(1,65)})
    for im_name, im_data in im_dic.items():
        print 'extracting from %s' % im_name
        slide_data = get_spot_data(im_data, grid_adjust[im_name])
        slide_sum = slide_data[['spot_number', 'sum']]
        slide_median = slide_data[['spot_number', 'median']]
        slide_sum.columns = ['spot_number', im_name + '_' + 'sum']
        slide_median.columns = ['spot_number', im_name + '_' + 'median']
        out_sum = pandas.merge(out_sum, slide_sum, how = 'inner', on = 'spot_number')
        out_median = pandas.merge(out_median, slide_median, how = 'inner', on = 'spot_number')
        # slide_data.to_csv(working_dir + now + '_' + im_name + '.csv')
    out_sum.to_csv(working_dir + now + '_raw_sum.csv', index = False)
    out_median.to_csv(working_dir + now + '_raw_median.csv', index = False)
    
    # plot alignment result
    print ''
    for im_name, im_data in im_dic.items():
        print 'plotting alignment for %s' % im_name
        fig = plt.figure()
        draw_array(im_data, grid_adjust[im_name])
        fig.savefig(working_dir + now + '_' + im_name + '_alignment.pdf')
        
    
if __name__ == '__main__':
    main()
        
        
    
