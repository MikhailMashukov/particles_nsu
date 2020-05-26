import cv2
import matplotlib.pyplot as plt
import json
import os
import io
import base64
import PIL
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from pycocotools import mask as maskUtils

# methods from labelMe project
# https://github.com/wkentaro/labelme/blob/master/labelme/utils/image.py

################
# data opening #
################

def img_b64_to_arr(img_b64):
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    img_arr = np.array(PIL.Image.open(f))
    return img_arr

def flip_v(img):
    h, w = img.shape[:2]
    flipped = np.zeros((h,w))
    for i in range(h):
        flipped[- i,:] = img[i,:]
        
    return flipped.astype(np.uint8)

def flip_h(img):
    """
    horizontal image flip
    :args: 
        :img: image to flip, numpy array
    :return:
        :flipped: flipped image, numpy array
    """
    
    h, w = img.shape[:2]
    flipped = np.zeros((h,w))
    for i in range(w):
        flipped[:,-i] = img[:,i]
    
    flipped = flipped.astype(np.uint8)
    
    return flipped

def open_height_map(path):
    """
    open file WSxM ASCII matrix file format
    :args:
        :path: path to WSxM file, str
    :return:
        :np_img: gray image as numpy array with normalized height in range[0..255], numpy array 
    """
    with open(path, 'r') as f:
        stp = f.read()
    
    first_img_byte = 132
    img = stp[first_img_byte:]

    data = []
    for line in stp.split('\n')[5:]:
        for num in line.split('\t'):
            try: 
                data.append(float(num))
            except:
                pass
    
    diff = 512**2 - len(data)
    if diff > 0:
        data += [0] * diff
        
    size = 512
    np_img = np.array(data).reshape(size, size)
    max_z = np.max(np_img)
    np_img = np_img * 255 / max_z
    np_img = np_img.astype(np.uint8)
    
    return np_img

def gray_map_to_bmp(gray_img):
    size = gray_img.shape[0]
    bmp = np.zeros((size, size, 3))
    for i in range(3):
        bmp[:,:,i] = gray_img
        
    return bmp.astype(np.uint8)

def resize_labelme_json(path_to_json):
    with open(path_to_json, 'rb') as f:
        data = json.load(f)
        
    img_b64 = data['imageData']
    img = img_b64_to_arr(img_b64)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    src_size = img.shape[0]
    print(img.shape)
    dst_size = 512
#    img = cv2.resize(img, (dst_size, dst_size))
    
    cnts = []
    for shape in data['shapes']:
        cnt = np.array(shape['points'])
#        cnt = cnt * dst_size / src_size
        cnts.append(cnt.astype(np.int32)) 
    
    return cnts, img

def open_resize_labelme_txt(WORK_DIR, name):
    json_name = name + '.json'
    path_to_json = os.path.join(WORK_DIR, json_name)
    
    cnts, img = resize_labelme_json(path_to_json)
    
    path_to_txt = r'C:\\Users\okune\Projects\STM\Raw/' + name + '.txt'
    inverted = open_txt(path_to_txt)
    
#    path_to_txt = r'C:\\Users\okune\Projects\STM\Raw/' + name + '.txt'
#    img = open_height_map(path_to_txt)

#    flipped = flip_v(img)
#    inverted = flip_h(flipped)
    
    return inverted, cnts

def open_txt(path_to_txt):
    
    img = open_height_map(path_to_txt)

    flipped = flip_v(img)
    inverted = flip_h(flipped)
    return inverted

##################################
# Prediction rendering utilities #
##################################

def get_contours(img, segms, inst_cls, inds):
    # contours extraction
    cnts = []
    print('Segms 0: ', segms[inst_cls][0])
    print(img.shape, np.mean(img), np.mean(img[0]))
    # print('inst_cls ', inst_cls, inds)
    for i in inds:
        blank = np.zeros(img.shape[:2])
        mask = maskUtils.decode(segms[inst_cls][i]).astype(np.bool)
        blank[mask] = 1
        contours, _ = cv2.findContours(blank.astype(np.uint8), cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        # if i == 0:
        #     # print('blank: ', blank.shape, blank.dtype)
        #     print(contours[0])
        cnts.append(contours[0])

    # remove too overlapping contours
    cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) != 0]
#    unique_cnts = remove_overlapping(img.shape[:2], cnts.copy())
    unique_cnts = remove_overlapping_masks(img.shape[:2], cnts.copy(), thresh = 0.5)
    print('Contours: {} predicted non zero area, {} unique'.format(len(cnts), len(unique_cnts)))
    
    return unique_cnts

def get_geometry(cnts):
    """
    calculate shape parameters of the particles and distance 
    between their centers with respect to their size 
    """
    moments = [cv2.moments(cnt) for cnt in cnts]
    areas = [max(M['m00'], 1) for M in moments] # to account for zero M['m00'] for non-zero contours
    # calculate position of mean weighted center of contour
    centers = [[int(M['m10']/area), int(M['m01']/area)] for M, area in zip(moments, areas)]
    centers = np.array(centers)
    # calculate sizes of particles
    part_sizes = [((int(area**0.5 ) + 1) // 2) * 2 for area in areas]
    # for all pairs of particles calculate distance between centers 
    distances = distance.cdist(centers, centers, 'euclidean')
    
    pair_sizes = [part_size + np.array(part_sizes) for part_size in part_sizes]
    # find mutual distances normalized per sum of particle sizes
    norm_distances = np.divide(distances, pair_sizes)
    for i in range(norm_distances.shape[0]):
        norm_distances[i,i] = 2
    
    return moments, areas, centers, part_sizes, norm_distances

def remove_overlapping_masks(mask_shape, cnts, thresh = 0.5):
    masks = []
    areas = []
    for cnt in cnts:
        mask = np.zeros(mask_shape)
        cv2.drawContours(mask, [cnt], -1, (1, 0, 0), -1)
        masks.append(mask)
        area = np.sum(mask)
        areas.append(area)
    inds = np.argsort(areas)
    
    unique_cnts = []
    unique_masks = []
    unique_areas = []
    for ind in inds:
        overlapped = False
        cnt = cnts[ind]
        mask = masks[ind]
        area = areas[ind]
        for unique_mask in unique_masks:
            overlap = np.where((mask + unique_mask) == 2, 1, 0)
            overlap_area = np.sum(overlap)            
            if overlap_area >= thresh * area:
                overlapped = True
        if not overlapped:
            unique_cnts.append(cnt)
            unique_masks.append(mask)
    
    return unique_cnts
            
    
def remove_overlapping(cnts, thresh = 0.3):
    """
    removes predictions were centers are closer
    than thresh * (sum of part sizes)
    """
    
    moments, areas, centers, part_sizes, norm_distances = get_geometry(cnts)   

    # on the other hand centers are far enough to be considered as a single particle
    twin_parts = np.where((0.01 <= norm_distances) & (norm_distances <= thresh))
    num_twin = len(twin_parts[0]) // 2
    
    twin_small = []
    for i in range(num_twin):
        if areas[twin_parts[0][i]] < areas[twin_parts[0][-1-i]]:
            twin_small.append(twin_parts[0][i])
        else:
            twin_small.append(twin_parts[0][-1-i])
            
    new_cnts = []
    for i, cnt in enumerate(cnts):
        if i not in twin_small:
            new_cnts.append(cnt)
    return new_cnts

def find_close_particles(cnts, thresh = 1.0):
    """"""
    moments, areas, centers, part_sizes, norm_distances = get_geometry(cnts)
    close_parts = np.where((0.1 <= norm_distances) & (norm_distances <= thresh))
    close_parts = np.array(close_parts).T
    close_parts = [np.sort(pair) for pair in close_parts]
    close_parts = np.array(close_parts)
    if close_parts.shape[0]:
        close_parts = np.unique(close_parts, axis = 0)
    
    return close_parts

def get_ext_cnts(cnts, ext_factor = 2):
    """"""
    moments, areas, centers, part_sizes, norm_distances = get_geometry(cnts)
    ext_cnts = []
    for i, cnt in enumerate(cnts):
        ext_cnt = (cnt - centers[i]) * ext_factor + centers[i]
        ext_cnts.append(ext_cnt.astype(np.int32))
    
    return ext_cnts

def get_line_params(centers, part_sizes):
    """
    returns line parameters (slope, bias) for line
    which goes through the point on the line from center[0]  to center[1]
    spaced proportionally to effective particle sizes 
    args:
        center: np. array of two centers of close lying particles
        part_sizes: corresponding sizes
    returns: slope, bias
    """
    dx, dy = centers[1] - centers[0]
    sep_point = centers[0] + np.array([dx, dy]) * part_sizes[0] / sum(part_sizes) 
    if dy != 0:
        slope = - dx/dy
    else:
        slope = - dx / 10**9

    bias = sep_point[1] - slope * sep_point[0]
    
    return slope, bias
    
def _line(point, slope, bias):
    """
    args:
        point: np.array of x, y coordinates
        slope: float 
    returns:
        y - slope * x - bias 
    """
    return point[1] - point[0] * slope - bias

def bound_cnt(cnt, center, slope, bias):
    """"""
    corr_cnt = []
    line_center = _line(center, slope, bias)
    sign_center = np.sign(line_center)
    for point in cnt:
        point = point.ravel()
        line_x_y = _line(point, slope, bias)
        sign_point = np.sign(line_x_y)
        if sign_center * sign_point >= 0:
            corr_cnt.append(point)
        else:
            dx, dy = point - center
            intersect = center - np.array([dx, dy]) * line_center / (line_x_y - line_center) 
            intersect = intersect.astype(np.int32)
            corr_cnt.append(intersect)
    
    return  np.array(corr_cnt).reshape(-1,1,2)  

def corr_ext_cnts_pair(ext_cnts_pair, centers, sizes_pair):
    """"""   
    slope, bias = get_line_params(centers, sizes_pair)
    
    corr_cnt_pair = []
    for cnt, center in zip(ext_cnts_pair, centers):  
        corr_cnt_pair.append(bound_cnt(cnt, center, slope, bias))
        
    return corr_cnt_pair  

def refine_ext_cnts(cnts, ext_cnts):
    # find particles at the distance <= some of their effective sizes
    close_parts = find_close_particles(cnts)

    print('Close parts: ', close_parts.shape)
    
    moments, areas, centers, part_sizes, norm_distances = get_geometry(cnts)

    # refine extended contours in order to take into account particles overlap
    for close_pair in close_parts:
        ind_0, ind_1 = close_pair[0], close_pair[1]
        ext_cnts_pair = [ext_cnts[ind_0], ext_cnts[ind_1]]
        centers_pair = [centers[ind_0], centers[ind_1]]
        sizes_pair = [part_sizes[ind_0], part_sizes[ind_1]]
        ext_cnts[close_pair[0]], ext_cnts[close_pair[1]] = corr_ext_cnts_pair(ext_cnts_pair,
                                                                          centers_pair,
                                                                          sizes_pair)
        
    return ext_cnts
    
###########################
# visualization utilities #
###########################

def display_pred(image):
    # convertion from opencv BGR format
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(num=1, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
    # Create figure and axes
    fig,ax = plt.subplots(1)
    fig.set_size_inches(18.5, 10.5)
    ax.axis('off')
    # Display the image
    ax.imshow(img)