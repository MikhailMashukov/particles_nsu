# Improved version of mmdet_nanopart.py.
# Is also called from the STMWeb site (the processFileForSite2*/3* methods)

from PIL import Image
import requests
from io import BytesIO

import mmcv
from mmcv import Config
from mmdet.apis import init_detector, inference_detector, show_result
import os
import numpy as np
import cv2
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt
from scipy.spatial import distance

# import labelme2coco
import coco2via_fit_predicted  # as coco2via     Let's import more straightforward yet
# from nano_stat import Nanostat
from utils import open_txt
from utils import remove_overlapping
#from utils import find_close_particles
from utils import get_ext_cnts
from utils import refine_ext_cnts
from utils import get_contours
#from utils import corr_ext_cnts_pair
from utils import display_pred
from ext_data_processing import *

color_mask = np.array((255, 0, 0)).astype(np.uint8)

def getModel3_0():
    config_file = 'cascade_mask_rcnn_x101_64x4d_fpn_1x_nanopart_3_0.py'
    checkpoint_file = 'weights/epoch_500_3x.pth'

    print('building model (config %s, weights %s)' % (config_file, checkpoint_file))
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # test a single image and show the results
    # print('model v. 3.0 initialized')
    return model

def processFileForSite3_0(srcFilePath, destImagePath, additOptions):
    print('Processing ', srcFilePath)
    srcImg = mmcv.imread(srcFilePath)
    if 0:
        stretchFactor = 3
        resized = cv2.resize(srcImg, None, fx=stretchFactor, fy=stretchFactor)
        img = resized #[:1215, :1215, :]
    else:
        stretchFactor = 1
        img = srcImg
    result = []
    res_bbox = []

    model = getModel3_0()
    result = inference_detector(model, img)
    bboxes, segms = result
    print('Bboxes 0: ', bboxes[0][0])
    inst_cls = 0
    inds = np.where(bboxes[inst_cls][:,-1] > 0.0001)[0]

    res_bbox = [bboxes[inst_cls][i].astype(np.float) for i in inds]

    # Visualizing masks and finding contours
    cnts = []
    predMasksImg = np.copy(img)
    for i in inds:
        mask = maskUtils.decode(segms[inst_cls][i]).astype(np.bool)      # E.g. 1215*1215 bool
        predMasksImg[mask] = predMasksImg[mask] * 0.6 + color_mask * 0.4

        blank = np.zeros(img.shape[:2])
        blank[mask] = 255
        contours, _ = cv2.findContours(blank.astype(np.uint8), cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            if len(contours) >= 2:
                for contourInd, contour in enumerate(contours):
                    print('Mask %d controur %d: %d point(s)' % (i, contourInd, len(contour)))
            cnts.append(contours[0])

    destBasePath, ext = os.path.splitext(destImagePath)

    imageInfo = { 'srcSize': srcImg.shape }
    if 'imageInfo' in additOptions:
        imageInfo.update(additOptions['imageInfo'])
    else:
        print('no image info')

    # display_pred(img, [])
    cv2.imwrite(destBasePath + '_Img_masks_predicted' + ext, predMasksImg)
    print('Predictions image file saved (%s)' % str(predMasksImg.shape))
    writeParticlesStatsCsv(destBasePath + '_Stat_bbox_predicted.csv', segms, imageInfo)

    res_img = cv2.drawContours(img, cnts, -1, (0, 255, 0), 1)
    cv2.imwrite(destBasePath + '.bmp', res_img)   # Predicted with green contours

    import utils

    img1Chan = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)

    img_url = additOptions['netSrcUrl']

    via_json = coco2via_fit_predicted.get_via_json(img, img_url, cnts, [])
    with open(destBasePath + '_Vis_VGG_predicted.json', 'w') as f:
        json.dump(via_json, f)
    print('VIA VGG project constructed for img {}'.format(img_url))

    # Writing images

    import psutil

    process = psutil.Process(os.getpid())
    print('Memory usage: ', [(n, v >> 20) for (n, v) in sorted(process.memory_info()._asdict().items())])

    # Printing several blocks of statistics and writing JSONs

    print('Resulting image: ', img.shape)
    jsonPath = destBasePath + '_Vis_Labelme_predicted.json'
    print(jsonPath)
    jsonImageName = os.path.basename(srcFilePath)
    writeParticlesContoursToJson(jsonPath, cnts, img, jsonImageName, 1)
    jsonPaths = [jsonPath]

    compMasks, imageSize = loadParticlesMasksFromJson(jsonPath)
    jsonImageInfo = { 'srcSize': imageSize }
    if 'imageInfo' in additOptions:
        jsonImageInfo.update(additOptions['imageInfo'])
    print('\nUnique masks (%s) statistics: ' % os.path.basename(jsonPath))
    printParticlesStats(None, [compMasks], jsonImageInfo)

if __name__ == '__main__':
    # Example
    processFileForSite3_0('sample/Pt-HOPG-01-0041.bmp', 'sample/Pt-Result.bmp',
            {'netSrcUrl': 'localhost/Pt-HOPG-01-0041.bmp'})       # Just something