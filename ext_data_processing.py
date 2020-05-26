# Mainly net results processing utils. Similar to data_processing.py in stmweb,
# but is located near nets and uses more libraries

import codecs
import io
import json
import math
import numpy as np
import os
import pandas
import pickle
import PIL
if os.path.abspath(__file__).lower().find('e:\\') < 0:      # Else - debugging on Mikhail's notebook without cv2 and pycocotools
    import cv2
    from pycocotools import mask as maskUtils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import traceback

import net_utils
# import labelme2coco
# import coco2via_fit_predicted  # as coco2via     Let's import more straightforward yet

def getIntParamFromAdditOptions(additOptions, paramName):
    if not paramName in additOptions:
        return None

    valStr = additOptions[paramName]
    if valStr.lower() == 'none':
        return None
    return int(valStr)

def getFloatParamFromAdditOptions(additOptions, paramName):
    if not paramName in additOptions:
        return None

    valStr = additOptions[paramName]
    if str(valStr).lower() == 'none':
        return None
    return float(valStr)


def display_pred(image, boxes, title=None, cmap=None, norm=None,
                   interpolation=None):
#     img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = image[:, :, [2, 1, 0]]      # BGR2RGB too
    plt.figure(num=1, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
    # Create figure and axes
    fig,ax = plt.subplots(1)
    fig.set_size_inches(18.5, 10.5)
    ax.axis('off')
    # Display the image
#     ax.imshow(img)
    if title is not None:
        ax.title(title, fontsize=9)

    for box in boxes:
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        xmin, ymin, w, h = xmin, ymin, xmax - xmin, ymax - ymin
        xmin, ymin, w, h = int(xmin), int(ymin), int(w), int(h)
        img[int(ymin):int(ymax), int(xmin):int(xmax), 2] += 150
#         rect = patches.Rectangle((xmin, ymin), w, h,linewidth=2,edgecolor='r',facecolor='none')
#         ax.add_patch(rect)
    ax.imshow(img)

def dumpResult(fileName, result):
    import pickle

    with open(fileName, 'wb') as file:
        pickle.dump(result, file)


# Swaps dimensions which are sometimes [height, width]
def getSwappedImageDimensions(img):
    shape = img.shape
    shape = tuple([shape[1], shape[0]] + list(shape)[2:])
    return shape

def encodeImageForJson(image):
    # Based on LabelMe's functions below
    img_pil = PIL.Image.fromarray(image, mode='RGB')
    f = io.BytesIO()
    img_pil.save(f, format='PNG')
    data = f.getvalue()
    encData = codecs.encode(data, 'base64').decode()
    encData = encData.replace('\n', '')
    return encData

# From LabelMe's sources,  https://github.com/wkentaro/labelme/blob/1d6ea6951c025a7db0540c7eac77577bc1507efa/labelme/utils/image.py#L10
# def img_b64_to_arr(img_b64):
#     f = io.BytesIO()
#     f.write(base64.b64decode(img_b64))
#     img_arr = np.array(PIL.Image.open(f))
#     return img_arr

# def img_arr_to_b64(img_arr):
#     img_pil = PIL.Image.fromarray(img_arr)
#     f = io.BytesIO()
#     img_pil.save(f, format='PNG')
#     img_bin = f.getvalue()
#     if hasattr(base64, 'encodebytes'):        # Generates \n and b' '
#         img_b64 = base64.encodebytes(img_bin)
#     else:
#         img_b64 = base64.encodestring(img_bin)
#     return img_b64


# Returns particles with sizes in specified range. Min and/or max can be None - no limit
def filterParticlesBySize(instBboxes, compMasks, minParticleSize, maxParticleSize):
    if minParticleSize is None and maxParticleSize is None:
        return instBboxes, compMasks

    filteredInstBboxes = []
    filteredCompMasks = []
    pi4sqrt = math.sqrt(math.pi / 4)
    for i, compMask in enumerate(compMasks):
        particleArea = maskUtils.area(compMask)
        particleDiam = math.sqrt(particleArea) / pi4sqrt
        if not minParticleSize is None:
            if particleDiam < minParticleSize:
                continue
        if not maxParticleSize is None:
            if particleDiam > maxParticleSize:
                continue
        filteredInstBboxes.append(instBboxes[i])
        filteredCompMasks.append(compMask)
    return filteredInstBboxes, filteredCompMasks


def _writeParticlesJsonHeader(outFile, srcImage, srcImageName):
    outFile.write('''{ "version": "3.16.1",
  "flags": {},
  "imagePath": "%s",
  "imageHeight": %d,
  "imageWidth": %d,
  "lineColor": [ 0, 255, 0, 128 ],
  "fillColor": [ 255, 0, 0, 128 ],
  "shapes": [\n''' % (srcImageName, srcImage.shape[0], srcImage.shape[1]))


# Writes masks to JSON in LabelMe's format. Contours (polygons) with area in pixels < minContourArea
# are not written
def writeParticlesMasksToJson(destFilePath, segms,
                       srcImage, srcImageName, coordsMult, minContourArea=0):
    problemMaskInds = set()
    with open(destFilePath, 'w') as outFile:
        _writeParticlesJsonHeader(outFile, srcImage, srcImageName)

        inst_cls = 0
        isFirstParticle = True
        for maskInd, compMask in enumerate(segms[inst_cls]):
            if maskUtils.area(compMask) <= 1:
                print('Warning: 0 or 1 pixel mask (index %d)' % maskInd)
                problemMaskInds.add(maskInd)
            mask = maskUtils.decode(compMask).astype(np.bool)
            maskImg = (mask * 1)
            maskImg = np.expand_dims(maskImg, 2).astype(np.uint8)
#             maskImg = np.tile(maskImg, (1, 1, 3))

#             return maskImg
            contours, _ = cv2.findContours(maskImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#             print('Found %d contours' % len(contours), _)
                # Can print like "Found 2 contours [[[ 1 -1 -1 -1]    [-1  0 -1 -1]]]",
            for contourInd, contour in enumerate(contours):
                # approx = cv2.approxPolyDP(contour, 0, True)
                #     # No documentation to approxPolyDP. The 2nd parameter is something like minimum
                #     # produced line length. 0.009 * cv2.arcLength(contour, True) produces about 9 points,
                #     # 0.005 * ... - 18, 0.003... - 27, 0.001... and less - 29

                if len(contour) < 3:
                    additInfo = ''
                    if len(contour) >= 1:
                        additInfo = ', near %s' % str(contour[0])
                    print('Warning: polygon with %d vertices (mask index %d, contour index %d / %d): %s. Skipping' % \
                          (len(contour), maskInd, contourInd, len(contours),
                           str(contour).replace('\n', ' ')))
                    problemMaskInds.add(maskInd)
#                     plt.imshow(mask);
                    continue

                area = cv2.contourArea(contour)
                if area < minContourArea:
                    print('Warning: polygon with area %.2f pixels (mask index %d, contour %d / %d): %s. Skipping' % \
                          (area, maskInd, contourInd, len(contours),
                           str(contour).replace('\n', ' ')))
                    problemMaskInds.add(maskInd)
                    continue
                elif len(contours) > 1:
                    # Printing information for all suspicious contours
                    print('Information: polygon area %.2f (mask %d, contour %d / %d)' %
                         (area, maskInd, contourInd, len(contours)))

                if isFirstParticle:
                    isFirstParticle = False
                else:
                    outFile.write(',\n')
                outFile.write('''    { "label": "nanoparticle %d%s",
      "line_color": null,
      "fill_color": null,
      "points": [\n''' % (maskInd, '' if contourInd == 0 else (', %d' % contourInd)))
                addition = '%8s' % ''

                for point in contour:
                    outFile.write('%s[ %.3f, %.3f ]' % (addition, point[0][0] * coordsMult, point[0][1] * coordsMult))
                    addition = ',\n%8s' % ''
                outFile.write('\n      ]\n    }')
#             break
        outFile.write('\n  ]')
        outFile.write(',\n  "imageData": "%s"\n}' % encodeImageForJson(srcImage))
    return problemMaskInds

def writeParticlesContoursToJson(destFilePath, cnts,
                       srcImage, srcImageName, coordsMult):
    with open(destFilePath, 'w') as outFile:
        _writeParticlesJsonHeader(outFile, srcImage, srcImageName)

        inst_cls = 0
        isFirstParticle = True
        for cntInd, cnt in enumerate(cnts):
            if len(cnt.shape) != 3 or cnt.shape[1:3] != (1, 2):
                raise Exception("Expected contours arrays of shape (*, 1, 2) (contours %d's shape: %s)" % \
                        (cntInd, str(cnt.shape)))

            if isFirstParticle:
                isFirstParticle = False
            else:
                outFile.write(',\n')
            outFile.write('''    { "label": "nanoparticle %d",
      "line_color": null,
      "fill_color": null,
      "points": [\n''' % (cntInd))
            addition = '%8s' % ''

            for point in cnt:
                outFile.write('%s[ %.3f, %.3f ]' % (addition, point[0][0] * coordsMult, point[0][1] * coordsMult))
                addition = ',\n%8s' % ''
            outFile.write('\n      ]\n    }')
#             break
        outFile.write('\n  ]')
        outFile.write(',\n  "imageData": "%s"\n}' % encodeImageForJson(srcImage))


def getAmplitude(imageInfo, axis):
    if axis == 0:
        fieldName = 'X Amplitude'
    else:
        fieldName = 'Y Amplitude'

    if not imageInfo or not fieldName in imageInfo:
        return None

    if imageInfo[fieldName][-2:] == ' m':
        v = float(imageInfo[fieldName][:-2])
    elif imageInfo[fieldName][-3:] == ' nm':           # TODO: microseconds? Need data to test
        v = float(imageInfo[fieldName][:-3]) * 1e-9
    else:
        return None
    return v

# Returns image size (width and height) in meters if there are enough metadata for this
def getImageMetersSize(imageInfo):
    imageMetersSize = (getAmplitude(imageInfo, 0), getAmplitude(imageInfo, 1))
    # print('imageMetersSize', imageMetersSize)
    if imageMetersSize[0] is None or imageMetersSize[1] is None:
        return None
    return imageMetersSize

def getPixelSize(imageInfo):       # TODO: better use image' size, not pixels' size - less error-prone
    imageMetersSize = getImageMetersSize(imageInfo)
    if imageMetersSize is None:
        return None
    return (imageMetersSize[0] / imageInfo['srcSize'][0], imageMetersSize[1] / imageInfo['srcSize'][1])

def getOutputImageStretchFactor(imageInfo, imageSize):
    # imageMetersSize = getImageMetersSize(imageInfo)
    if imageInfo is None:
        return None
    stretchFactor = float(imageSize[0]) / imageInfo['srcSize'][0]
    if abs(stretchFactor - float(imageSize[1]) / imageInfo['srcSize'][1]) > 1e-10:
        print('Warning: non-square image stretching')

    if abs(stretchFactor - round(stretchFactor)) < 1e-10:
        return round(stretchFactor)
    else:
        return stretchFactor

# Does some checks of masks and prints some basic statistics for them
def printBaseParticlesStats(bboxes, segms, imageInfo):
    inst_cls = 0
    nonEmptyInstClsCount = 0
    for segm in segms:
        if segm:
            nonEmptyInstClsCount += 1
    print('Groups (instance classes) in prediction: %d, non-empty: %d' % \
            (len(segms), nonEmptyInstClsCount))
    if nonEmptyInstClsCount > 1:
        print('Warning: statistics only includes the first one')

    particleCount = len(segms[inst_cls])
    mainInfo = []
    if bboxes is not None and bboxes[inst_cls].shape[0] != len(segms[inst_cls]):
        print('Error: bboxes and masks number mismatch (%d and %d)' % \
                (bboxes[inst_cls].shape[0], len(segms[inst_cls])))
    else:
        mainInfo.append('Detected particles: %d' % particleCount)

    # print('Segms: ', len(segms[inst_cls]), segms[inst_cls][0])   #d_
    imageSize = segms[inst_cls][0]['size']       # Mask's, maybe stretched image size
    for compMask in segms[inst_cls]:
        if imageSize != compMask['size']:
            print('Error: different masks sizes')
            return
    print('Output image size: %s pixels' % (str(imageSize)))

    return mainInfo

# Calculates and prints main statistics for the set of particles.
# ImageInfo can be None, it is only used to get physical dimensions of the picture,
# statistics in pixels will be calculated without it.
# Can be used on cropped images. In that case
#   * segms contain not cropped masks, but only those falling into the cropped area,
#   * imageInfo is empty or contain information about entire image,
#   * cropAreaPart contain cropped area/source area ratio (e.g. if border is 100 pixels at each size and
#     and image size is 1000 * 1000, cropAreaPart will be (1000 - 200) ** 2 == 0.64)
def printMainParticlesStats(segms, imageInfo,
                            mainInfoToPrint, cropAreaPart = 1,
                            printLineBeforeMainInfo = True):
    inst_cls = 0
    mainInfo = mainInfoToPrint               # List into which messages which should be after ------- are saved
    particleCount = len(segms[inst_cls])
    if particleCount == 0:
        print('No particles')
        return
    imageSize = segms[inst_cls][0]['size']   # Mask's, maybe stretched image size

    pixelSize = None
    stretchFactor = None
    if imageInfo:
        pixelSize = getPixelSize(imageInfo)
        stretchFactor = getOutputImageStretchFactor(imageInfo, imageSize)
        if pixelSize:
            pixelSize = [pixelSize[0] / stretchFactor, pixelSize[1] / stretchFactor]
            print('Masked image pixel size: ', pixelSize)

    particleAreaSum = 0     # In masks' pixels (at e.g. 1215 * 1215 image)
    particleAreaSqrtSum = 0
    particlesImagePart = 0
    if particleCount != 0:
        particleAreas = [maskUtils.area(compMask) for compMask in segms[inst_cls]]
        for particleArea in particleAreas:
            particleAreaSum += particleArea
            particleAreaSqrtSum += math.sqrt(particleArea)
        avgParticleArea = particleAreaSum / particleCount
        mess = 'Average particle area: %.1f pixels' % (avgParticleArea)
        if pixelSize:
            mess += ', %.3f nm^2' % (avgParticleArea * \
                    pixelSize[0] * pixelSize[1] * 1e18)
        print(mess)

        unitedCompMask = maskUtils.merge(segms[inst_cls], intersect=False)
        unitedMaskPixelCount = maskUtils.area(unitedCompMask)
        print('Total particles area: %d pixels' % unitedMaskPixelCount)
        print('Overlapping area: %d pixels' % (particleAreaSum - unitedMaskPixelCount))
        particlesImagePart = float(unitedMaskPixelCount) / (imageSize[0] * imageSize[1] * cropAreaPart)
        mainInfo.append('Particles area: %.3f%%' % (particlesImagePart * 100))

        pi4sqrt = math.sqrt(math.pi / 4)
        # Среднеповерхностный диаметр. Area = pi * r^2 = pi * d^2 / 4; d^2 = area / pi * 4
        midSurfaceDiam = math.sqrt(avgParticleArea) / pi4sqrt
        # Средний проектированный диаметр,  Pi pixels area will mean 1-pixel radius
        avgProjectedAreaDiam = particleAreaSqrtSum / particleCount / pi4sqrt
        midSum = 0
        projectedSum = 0
        for particleArea in particleAreas:
            diam = math.sqrt(particleArea) / pi4sqrt
            midSum += (diam - midSurfaceDiam) ** 2
            projectedSum += (diam - avgProjectedAreaDiam) ** 2
        midSurfaceDiamStdDev = math.sqrt(midSum / particleCount)
        projectedAreaDiamStdDev = math.sqrt(projectedSum / particleCount)

        mess = 'Mid-surface diameter: %.2f pixels' % midSurfaceDiam
        if pixelSize:
            mess += ', %.4f nm' % \
                    (midSurfaceDiam * (pixelSize[0] + pixelSize[1]) / 2 * 1e9)
                # Actually not correct for non-square pixels, we should analyze source pixels
                # with respect to x and y axes instead of simple particleAreaSqrtSum += math.sqrt(particleArea)
        mainInfo.append(mess)
        mess = 'Mid-surface diameter std. dev.: %.3f pixels' % midSurfaceDiamStdDev
        if pixelSize:
            mess += ', %.5f nm' % \
                    (midSurfaceDiamStdDev * (pixelSize[0] + pixelSize[1]) / 2 * 1e9)
        mainInfo.append(mess)

        mess = 'Average projected area diameter: %.2f pixels' % avgProjectedAreaDiam
        if pixelSize:
            mess += ', %.4f nm' % \
                    (avgProjectedAreaDiam * (pixelSize[0] + pixelSize[1]) / 2 * 1e9)
#                     (avgParticleDiameter * math.sqrt(pixelSize[0] ** 2 + pixelSize[1] ** 2) * 1e9)
        mainInfo.append(mess)
        mess = 'Average projected area diameter std. dev.: %.3f pixels' % projectedAreaDiamStdDev
        if pixelSize:
            mess += ', %.5f nm' % \
                    (projectedAreaDiamStdDev * (pixelSize[0] + pixelSize[1]) / 2 * 1e9)
        mainInfo.append(mess)

        mess = 'Density: %.4f particles / 1000 pixels' % (particleCount * 1000.0 / \
                (imageSize[0] * imageSize[1] * cropAreaPart))
        imageMetersSize = getImageMetersSize(imageInfo)
        if imageMetersSize:
            mess += ', %.4f / nm^2' % (particleCount /
                    (imageMetersSize[0] * imageMetersSize[1] * cropAreaPart * 1e18))
        mainInfo.append(mess)

    if printLineBeforeMainInfo:
        print('--------------------')
    for line in mainInfo:
        print(line)

# Old method that calculates and prints full statistics about particles in the form of masks.
# ImageInfo can be None, it is only used to get physical dimensions of the picture,
# statistics in pixels will be calculated without it.
def printParticlesStats(bboxes, segms, imageInfo,
                        printLineBeforeMainInfo = True):
    mainInfoToPrint = printBaseParticlesStats(bboxes, segms, imageInfo)
    if bboxes is not None:
        inst_cls = 0
        inds = np.where(bboxes[inst_cls][:,-1] > 0.0001)[0]
        print('Degenerated bboxes: %d' % (len(bboxes[inst_cls]) - len(inds)))

    printMainParticlesStats(segms, imageInfo,
                            mainInfoToPrint, 1, printLineBeforeMainInfo)

def calcParticlesStatsForCsv(segms, imageInfo):
    # Somewhat similar to printParticlesStats. Checks are mainly made there, the code here is simpler

    inst_cls = 0
    particleCount = len(segms[inst_cls])
    data = { 'index': [], 'bbox_center_x': [], 'bbox_center_y': [],
             'area_pixels': [], 'diameter_pixels': [] }      # TODO: max height, min height, center of mass...
    if particleCount == 0:
        return data

    imageSize = segms[inst_cls][0]['size']       # Mask's, maybe stretched, image size
    pixelSize = getPixelSize(imageInfo)
    if pixelSize:
        stretchFactor = getOutputImageStretchFactor(imageInfo, imageSize)
        pixelSize = [pixelSize[0] / stretchFactor, pixelSize[1] / stretchFactor]
        print('Csv pixelSize ', pixelSize)

        data['area_nm2'] = []
        data['diameter_nm'] = []

    for maskInd, compMask in enumerate(segms[inst_cls]):
        data['index'].append(maskInd)
        bbox = maskUtils.toBbox(compMask)   # Bbox here is (x0, y0, width, height)
        # if maskInd < 7:
        #     print('CSV mask bbox ', bbox)
        data['bbox_center_x'].append(bbox[0] + bbox[2] / 2.0)     # In output image's pixels
        data['bbox_center_y'].append(bbox[1] + bbox[3] / 2.0)
        particleArea = maskUtils.area(compMask)
        data['area_pixels'].append(particleArea)
        particleDiam = math.sqrt(particleArea / math.pi * 4)
        data['diameter_pixels'].append(particleDiam)

        if not pixelSize is None:
            data['area_nm2'].append(particleArea * pixelSize[0] * pixelSize[1] * 1e18)
            data['diameter_nm'].append(particleDiam * (pixelSize[0] + pixelSize[1]) / 2 * 1e9)
    return data

def writeParticlesStatsCsv(destFilePath, segms, imageInfo, saveCommasCopy=True,
                           isInsideCroppedAreaFlags=None):
    data = calcParticlesStatsForCsv(segms, imageInfo)
    if isInsideCroppedAreaFlags is None:
        df = pandas.DataFrame(data)
    else:
        data['is_at_border'] = [(0 if isInsideCropped else 1) for isInsideCropped in isInsideCroppedAreaFlags]
        df = pandas.DataFrame(data)
        cols = df.columns.tolist()
        assert len(cols) >= 4
        cols[3], cols[-1] = cols[-1], cols[3]
        df = df[cols]
    df.to_csv(destFilePath, sep=';', decimal='.', index=False)
    if saveCommasCopy:
        path, ext = os.path.splitext(destFilePath)
        destFilePath_commas = path + '_Commas' + ext
        df.to_csv(destFilePath_commas, sep=';', decimal=',', index=False)
    return data

# df = pandas.DataFrame({'name': ['Raphael', 'Donatello'],
# ...                    'mask': ['red', 'purple'],
# ...                    'weapon': ['sai', 'bo staff']})
# >>> df.to_csv(index=False)


def loadParticlesMasksFromJson(jsonPath):
    with open(jsonPath, 'r') as file:
        jsonStr = file.read()
    j = json.loads(jsonStr)

    imageSize = (j['imageWidth'], j['imageHeight'])
    stretchFactor = 3   # # We'll render to e.g. (415 * 6) * (415 * 6). Should be nice after initial, net's, stretch of 2 or 3
                        # With 6 and initial 1000 * 1000 JSON can be slow (30 seconds for one JSON), so let's set 3
    compMasks = []
    for particleInd, particleJson in enumerate(j['shapes']):
        # We could calculate area of each polygon here, but not the area of theirs intersection.
        # So making masks (this also gives possibility to run existing code in printParticlesStats)

        image = np.zeros((imageSize[0] * stretchFactor, imageSize[1] * stretchFactor), dtype=np.uint8)
            # With , order='F' cv2.drawContours doesn't work
        points = particleJson['points']
        points = np.array(points) * stretchFactor
        points = np.round(points).astype(np.int32)
        cv2.drawContours(image, [points], -1, 255, -1)
        # print('drawContours')               # With print = print_timeMeasure in net_utils.py this can show execution time.
        a = np.asfortranarray(image)          # It can be about 0.17 seconds per call for 6000 * 6000 image here
        # print('np.asfortranarray(image)')
        compMask = maskUtils.encode(a)        # and 0.05 s here. Other operations - about 0.001 s
                                              # For 3000 * 3000 these times become 0.01 - 0.03 and 0.013
        # print('encoded (%d bytes)' % (len(compMask['counts'])))
        maskArea = maskUtils.area(compMask)
        if maskArea < 5:
            print('Warning: particle %d area is %.1f pixel(s)' % (particleInd, maskArea))

        if 0:   #d_
            pi4sqrt = math.sqrt(math.pi / 4)
            particleArea = maskUtils.area(compMask)
            particleDiam = math.sqrt(particleArea) / pi4sqrt
            if particleDiam < 10:
                print('Particle %d area %f. JSON points min x ' % (particleInd, particleArea),
                      points[:, 0].min(),
                      ', max y ', points[:, 0].max(), 'minY ', points[:, 1].min(),
                      ', max y ', points[:, 1].max())
                print(points)

            bbox = maskUtils.toBbox(compMask)
            print('mask extent: ', bbox)
#             if bbox[2] == bbox[3]:
#                 print('%d. ' % particleInd, particleJson['points'])
        compMasks.append(compMask)
    return compMasks, imageSize

# This method is called from image_processing_form. Displays text to show, prints data for histogram
# and generates .csv that is usually absent for old data (up to December 2019).
# JsonPath - from result or (if LabelMe JSON was loaded as a source) - source JSON itself
# TODO: currently auto-generated <source JSON>_coco/via.json are generated, but not visible. Need to pass
# path to a last result, to attach it there
def processJsonParticlesStats(jsonPath, imageInfo, destCsvPath=None, printHistogramData=True):
    print('Image info: ', net_utils.cutLongImageInfo(imageInfo))

    compMasks, imageSize = loadParticlesMasksFromJson(jsonPath)
    jsonImageInfo = { 'srcSize': imageSize }
    if imageInfo:
        jsonImageInfo.update(imageInfo)
        # Printing freshly calculated statistics just in case.
        # Previous logs can contain incorrect sizes in nm

    destCocoJsonPath = '%s_Coco.json' % os.path.splitext(jsonPath)[0]
    print('Writing COCO JSON to %s' % destCocoJsonPath)
    labelme2coco.labelme2coco([jsonPath], destCocoJsonPath)

    destViaJsonPath = '%s_Via.json' % os.path.splitext(jsonPath)[0]
    print('Writing VIA JSON to %s' % destViaJsonPath)
    coco2via_fit_predicted.main(destViaJsonPath, destCocoJsonPath)

    if destCsvPath is None:
        destCsvPath = '%s_Particles.csv' % os.path.splitext(jsonPath)[0]
            # TODO? This hardcoded _Particles.csv addition is present in many places. It could be e.g. in net_utils.py
            # (because it is already linked between STMWeb and Nets). But for now importing from it will only
            # increase the mess. Let's keep until necessity to change or general code sorting-out
    if not os.path.exists(destCsvPath) and \
       not os.path.exists('%s_CorrParticles.csv' % os.path.splitext(jsonPath)[0]):
        print('Writing particles data to ', destCsvPath)
        csvData = writeParticlesStatsCsv(destCsvPath, [compMasks], jsonImageInfo)
    else:
        csvData = calcParticlesStatsForCsv([compMasks], jsonImageInfo)

    printParticlesStats(None, [compMasks], jsonImageInfo)

    if printHistogramData:
        if 'diameter_nm' in csvData:
            histogramSrcData = csvData['diameter_nm']
        else:
            histogramSrcData = csvData['diameter_pixels']

        # print(histogramSrcData)    #d_
        print('HistogramData: ', net_utils.encodeToStr(histogramSrcData))
            # Returning data for histogram separately. Accepting module knows that such string
            # needs to be decoded and displayed graphically
    return compMasks

# Does what is needed for JSON, edited by user in LabelMe and uploaded to the site:
# the same as processJsonParticlesStats and also writing of image with masks
def processLabelMeJson(jsonPath, destCsvPath, destImagePath):
    compMasks = processJsonParticlesStats(jsonPath, None, destCsvPath, False)

    color_mask = np.array((255, 0, 0)).astype(np.uint8)   # BGR for cv2

    if not compMasks:
        return
    # imageSize = compMasks[0]
    image = None
    for maskInd, compMask in enumerate(compMasks):
        mask = maskUtils.decode(compMask).astype(np.bool)
        # maskImg = (mask * 1)
        # print('processLabelMeJson mask size %d' % maskInd, mask.shape)
        # maskImg = maskImg # np.expand_dims(maskImg, 2).astype(np.uint8)
        if image is None:
            image = np.ones(list(mask.shape) + [3], dtype=np.uint8) * 128
        image[mask] = image[mask] * 0.6 + color_mask * 0.4
    cv2.imwrite(destImagePath, image)


class CropParticlesMasksFilter:
    # Img shape here is [height, width, channels]
    def __init__(self, img, segms):
        csvData = calcParticlesStatsForCsv(segms, {})
            # We only need average particles diameter, but it's desirable to reuse existing code.
            # Giving no image info in order not to confuse ourselves with unstretching

        borderSize = 0
        if csvData['diameter_pixels']:
    #         print(csvData['diameter_pixels'])
            self.avgProjectedAreaDiam = sum(csvData['diameter_pixels']) / len(csvData['diameter_pixels'])
            borderSize = (self.avgProjectedAreaDiam + 1.9) // 2
        self.borderSize = borderSize
        imgShape = getSwappedImageDimensions(img)
        self.croppedImgBbox = ((borderSize, borderSize), (imgShape[0] - borderSize, imgShape[1] - borderSize))

    def isParticleInsideCroppedArea(self, bbox):
        assert bbox[2] >= bbox[0] and bbox[3] >= bbox[1]         # bboxes here - (x0, y0, xk, yk)
        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2
        return self.croppedImgBbox[0][0] <= x and x < self.croppedImgBbox[1][0] and \
               self.croppedImgBbox[0][1] <= y and y < self.croppedImgBbox[1][1]


def printUsage():
    print("Run format: python %s.py <encoded params>" %
          (__name__))

if __name__ == "__main__":
    # print('Arguments: ', sys.argv)

    import codecs
    import pickle

    if len(sys.argv) == 2:
      try:
        encParams = sys.argv[1]

        # module = __import__('foo')
        # func = getattr(module, 'bar')
        # func()
        params = net_utils.decodeObject(encParams)
        print(net_utils.cutLongStrings(params))
        locals()[params['methodName']](*params['args'])
      except BaseException as e:
        print('Exception in ext_data_processing.py: ', str(e))
        try:
            # tbInfo = ''.join(traceback.format_stack())
            type, value, tb = sys.exc_info()
            print(traceback.format_tb(tb))
        except:
            pass
      # exit(0)
    else:
        printUsage()
