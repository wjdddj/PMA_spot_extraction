#!/Users/jwang/anaconda/bin/

import matplotlib.pyplot as plt
import numpy as np
import pandas
import os
import re
from time import gmtime, strftime

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

def adjust_grid(grid, x_shift, y_shift, res):
    grid['grid X'] = grid['grid X']/res + x_shift
    grid['grid Y'] = grid['grid Y']/res + y_shift
    grid['grid X'] = grid['grid X'].astype(int)
    grid['grid Y'] = grid['grid Y'].astype(int)
    return grid   

def get_spot(im_data, centroid, radius):
    x1 = int(centroid[0] - radius)
    x2 = int(centroid[0] + radius)
    y1 = int(centroid[1] - radius)
    y2 = int(centroid[1] + radius)
    spot = im_data[y1:y2, x1:x2]
    return spot

def get_spot_data(im_data, grid):
    now = strftime("%Y%m%d", gmtime())
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

def draw_box(im_data, centroid, radius):
    x1 = centroid[0] - radius
    x2 = centroid[0] + radius
    y1 = centroid[1] - radius
    y2 = centroid[1] + radius
    plt.plot((x1, x2), (y1, y1), color='r', linewidth = 0.2)
    plt.plot((x1, x2), (y2, y2), color='r', linewidth = 0.2)
    plt.plot((x1, x1), (y1, y2), color='r', linewidth = 0.2)
    plt.plot((x2, x2), (y1, y2), color='r', linewidth = 0.2)
    return

def draw_array(im_data, grid):
    plt.imshow(np.log(im_data), cmap = 'Greys')
    plt.axis('off')
    for i in range(0,64):
        draw_box(im_data, (grid['grid X'][i], grid['grid Y'][i]), 250)
    return


def main():
    
    working_dir = os.getcwd() + '/'
    now = strftime("%Y%m%d", gmtime())
    
    # prepare grid file
    res = 5
    x_shift = 0
    y_shift = 25
    grid = pandas.read_csv(working_dir + 'grid_1.csv')
    grid = adjust_grid(grid, x_shift, y_shift, res)
    grid.to_csv(working_dir + now + '_grid_out.csv')
    
    # read in images
    im_dic = read_images(working_dir)

    # obtain data based on aligned grid
    print ''
    out_sum = out_median = pandas.DataFrame({'spot_number':range(1,65)})
    for im_name, im_data in im_dic.items():
        print 'extracting from %s' % im_name
        slide_data = get_spot_data(im_data, grid)
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
        draw_array(im_data, grid)
        fig.savefig(working_dir + now + '_' + im_name + '_alignment.pdf')
    
if __name__ == '__main__':
    main()
        
        
    
