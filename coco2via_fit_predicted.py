# Coco to Via (VGG Image Annotator) JSON converter.
# Similar to coco2via.py, but adjusted for existing categories

import json
import labelme2coco

def load_annotations(json_path):

    with open(json_path) as f:
        annotations = json.load(f)

    return annotations

# conversion to bounding box from points
def polygon_to_bbox(pts):
    import math
    
    xmin = math.inf
    xmax = -math.inf
    ymin = math.inf
    ymax = -math.inf
    for i in range(0,len(pts),2):
    
        if ( pts[i] > xmax ):
            xmax = pts[i]
        if ( pts[i] < xmin ):
            xmin = pts[i]
        if ( pts[i+1] > ymax ):
            ymax = pts[i+1]
        if ( pts[i+1] < ymin ):
            ymin = pts[i+1]
            
    return [xmin, ymin, xmax-xmin, ymax-ymin]

def coco2via(coco_json):
    
    # load original via data
    output = {}
    coco = load_annotations(coco_json)

    # create an index of all categories
    category_list = {}
    s = coco['categories'][0]['name']
    category_list = ''.join([i for i in s if not i.isdigit()]) # nanoparticle
    
    # create an index of all annotations
    annotation_list = {}
    for annotation_index in coco['annotations']:
        coco_image_id = annotation_index['image_id']
        if not coco_image_id in annotation_list:
            annotation_list[coco_image_id] = []
        annotation_list[coco_image_id].append(annotation_index)
    
    size = 0
    # add all files and annotations
    for coco_img_index in coco['images']:
        filename = coco_img_index['file_name']
        #if 'coco_url' in coco_img_index:
        #    filename = coco_img_index['coco_url']

        via_img_id = filename
        coco_img_id = coco_img_index['id']
        width = coco_img_index['width']
        height = coco_img_index['height']
        if size == 0:
            output[via_img_id] = {'filename':filename,
                                'size':size,
                                'regions':[],
                                'file_attributes':{'width':width, 'height':height},
                                };
        size += 1
        # add all annotations associated with this file
        if coco_img_id in annotation_list:
            for i in annotation_list[coco_img_id]:
                annotation = i
                bbox = polygon_to_bbox(i['segmentation'][0])
                area = bbox[2] * bbox[3]
                r = { 'shape_attributes': { 'name':'polygon', 'all_points_x':[], 'all_points_y':[] },
                      'region_attributes': {},
                    }

                # fix for variations in segmentation:
                # annotation['segmentation'] = [x0,y0,x1,y1,...]
                # annotation['segmentation'] = [[x0,y0,x1,y1,...]]
                seg = annotation['segmentation']
                if ( len(seg) == 1 and len(seg[0]) != 0 ):
                    seg = annotation['segmentation'][0]
            
                for j in range(0,len(seg),2):
                    r['shape_attributes']['all_points_x'].append(seg[j])
                    r['shape_attributes']['all_points_y'].append(seg[j+1])
                    
                cat_name = category_list
                r['region_attributes']['category'] = cat_name
                if coco_img_id == 0:
                    r['region_attributes'][cat_name] = 'predicted'
                if coco_img_id == 1:
                    r['region_attributes'][cat_name] = 'fit'
                output[via_img_id]['regions'].append(r)
            
    return output, cat_name

def get_via_json(img, img_url, pred_cnts, fit_cnts):
    via_json = {}   
    
    # write '_via_img_metadata' values
    meta = {}

    # add file_attributes
    h, w = img.shape[:2]
    meta['file_attributes'] = dict( 
                                    width = w,
                                    height = h
                                    )
    
    # add predicted contours 
    regs = []
    for cnt in pred_cnts:
        reg = {}
        reg['shape_attributes'] = dict(
                                    name = 'polygon',
                                    all_points_x = cnt.T[0].ravel().tolist(),
                                    all_points_y = cnt.T[1].ravel().tolist()
                                    )
        reg['region_attributes'] = dict( category = 'nanoparticle',
                                         nanoparticle = 'predicted'
                                       )
        regs.append(reg)
    # add fitted contours 
    for cnt in fit_cnts:
        reg = {}
        reg['shape_attributes'] = dict(
                                    name = 'polygon',
                                    all_points_x = cnt.T[0].ravel().tolist(),
                                    all_points_y = cnt.T[1].ravel().tolist()
                                    )
        reg['region_attributes'] = dict( category = 'nanoparticle',
                                         nanoparticle = 'fitted'
                                       )
        regs.append(reg) 
    
    meta['regions'] = regs
    
    # add size, name
    meta['filename'] = img_url
    meta['size'] = -1
    
    meta_key = img_url + '-1'
    via_json['_via_img_metadata'] = {meta_key : meta}  
    
    # add _via_settings
    via_json['_via_settings'] = {'ui': {'image': {'region_color': 'nanoparticle'}}}
    
    # add _via_attributes
    via_json['_via_attributes'] = { 'region': {'nanoparticle': {'nanoparticle': 'text',
                                                                'type': 'radio',
                                                                'options': { 'fitted': 'particle contours, fitted with Gauss surface',
                                                                                'predicted': 'particle contours, predicted by Neural Network'
                                                                          },
                                                                'default_options': {'fitted': True}
                                                                }
                                              },
                                    'file': {'width': {'type': 'text'}, 'height': {'type': 'text'}}
                                  }
    
    return via_json

if __name__ != '__main__':
    def main(output_path, input_path):
        import os
        output, cat_name = coco2via(input_path)
        # formatting json to look like via project file, creating two categories of objects for fitted and predicted by NN
        data = {}
        data['_via_img_metadata'] = output
        data['_via_settings'] = {'ui': {'image':{'region_color': cat_name}}}
        data['_via_attributes'] = {"region": 
                                   {cat_name: 
                                    {cat_name: "text", 
                                     'type': 'radio', 
                                     'options':{
                                         'fit': 'Regions, normalized with Gaussian surface',
                                         'predicted': 'Regions, predicted by Neural Network'},
                                     'default_options': {'fit': True}
                                    }},
                                   "file":
                                   {"width":
                                    {"type": "text"},
                                    "height": 
                                    {"type": "text"}}}

        # writing final file to <output_path>/via_converted.json
        # if not os.path.exists(output_path):
        #     os.mkdir(output_path)
        #     print('Folder didn\'t exist, created new one')
        # with open(os.path.join(output_path, 'via_converted.json'), 'w') as f:
        #     print('Saved VIA project JSON')
        #     print(os.path.join(output_path, 'via_converted.json'))

        # output_path is final file's path
        with open(output_path, 'w') as f:
            json.dump(data, f)
