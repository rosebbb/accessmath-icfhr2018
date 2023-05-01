import numpy as np
# from utils import mask2poly
import json
from pycocotools.coco import COCO
import os
import cv2
import copy
import sys
import random
# from maskpoly_tools import mask2poly

'''
__init__
load_json
json_info
remove_ann
dump_json
create_cat_entry
get_cat_names
get_cat_map
cat_stat
extract_json
remove_repeat_instance
get_defective_imgs
get_defect_free_imgs
get_anns
remove_non_defective_image
change_img_extension
change_file_name
crop_out_encap
sanity_check
save_instance
combine_json
combine_to_json
addPred2json
partition_dataset
get_info
create_img_entry
create_ann_entry
'''

class Ann_utils():
    def __init__(self, json_file, img_dir=None):
        self.img_dir = img_dir
        self.json_file = json_file

        if os.path.isfile(json_file) and os.access(json_file, os.R_OK):
            self.load_json()

    def load_json(self):
        with open(self.json_file) as j:
            data = json.load(j)
        cocoapi = COCO(self.json_file)
    
        self.data = data
        self.cocoapi = cocoapi
        return self.data, self.cocoapi

    # get info of the json
    def json_info(self, verbose=False):
        # categories
        cats_stat = self.cat_stat()
        cat_id_name_map, cat_name_id_map = self.get_cat_map()

        if verbose:
            print(f'\n- number of instances in each category in  {os.path.basename(self.json_file)} -')
            for cat_id in cats_stat:
                print(cat_id_name_map[cat_id].ljust(20), ' has ', str(cats_stat[cat_id]).ljust(10), ' instances')
            print('----------------------')

        # number of images without annotations
        images_no_ann = []
        images_ann = []
        img_name_id_map = {}
        for image_info in self.data['images']:
            image_file = image_info['file_name']
            img_name_id_map[image_file] = image_info['id']
            ann_ids = self.cocoapi.getAnnIds(image_info['id'])
            # anns = cocoapi.loadAnns(ann_ids)

            if len(ann_ids) == 0: # need to double check
                images_no_ann.append(image_file)
            else:
                images_ann.append(image_file)

        # number of images and anns
        if verbose:
            print('number of images:', len(self.data['images']))
            print('number of annotations:', len(self.data['annotations']))
            print('number of images without annotations', len(images_no_ann))

        # get max ann id
        ann_ids = [ann['id'] for ann in self.data['annotations']]
        max_ann_id = max(ann_ids)

        # get max img id
        img_ids = [img['id'] for img in self.data['images']]
        max_img_id = max(img_ids)
        print('max(img_ids)', max_img_id)

        ann_entry = self.data['annotations'][0]
        img_entry = self.data['images'][0]

        return max_ann_id, max_img_id, img_name_id_map, ann_entry, img_entry

    def get_imgid(self, img_name):
        for img_info in self.data['images']:
            if img_info['file_name'] == img_name:
                return img_info['id']

        return None

    # Clean annotations
    def remove_ann(self, cats2remove=None, cats2keep=None):
        '''
        Remove unused annotations
        '''
        data = self.data
        _, cat_name_id_map = self.get_cat_map()
        cat_stat = self.cat_stat()
        print('cat_stat', cat_stat)

        print('number of annotations before: ', len(data['annotations']))
        print('number of images before: ', len(data['images']))
        print('categories before: ', data['categories'])

        if cats2remove is not None:
            cats_info = [c for c in data['categories'] if c['name'] not in cats2remove]
            data['categories'] = cats_info
            print('categories after: ', data['categories'])

            catids2remove = [cat_name_id_map[c] for c in cats2remove]
            anns_info = [a for a in data['annotations'] if a['category_id'] not in catids2remove]  # need to double check xing
            data['annotations'] = anns_info

        if cats2keep is not None:
            cats_info = [c for c in data['categories'] if c['name'] in cats2keep]
            cats2keep_id = [cat_name_id_map[c] for c in cats2keep]

            data['categories'] = cats_info
            print('categories after: ', data['categories'])

            anns_info = [a for a in data['annotations'] if a['category_id'] in cats2keep_id]  # need to double check xing
            data['annotations'] = anns_info

        print('number of annotations after: ', len(data['annotations']))
        print('number of images after', len(data['images']))
        return data

    # dump json
    def dump_json(self, data):
        print('dumping to ', self.json_file)
        with open(self.json_file, 'w') as j:
            json.dump(data, j)

    def create_cat_entry(self, id, name):
        cat_entry = {}
        cat_entry['id'] = id
        cat_entry['name'] = name
        cat_entry['supercatory'] = ''
        return cat_entry

    def get_cat_names(self):
        with open(self.json_file) as j:
            data = json.load(j)
        cat_names = [cat_info['name'] for cat_info in data['categories']]

        return cat_names   

    def get_cat_map(self):
        cat_id_name_map = {}
        cat_name_id_map = {}
        for cat_info in self.data['categories']:
            cat_id_name_map[cat_info['id']] = cat_info['name']
            cat_name_id_map[cat_info['name']] = cat_info['id']

        return cat_id_name_map, cat_name_id_map

    def cat_stat(self):
        cat_stat = {}
        for cat_info in self.data['categories']:
            cat_stat[cat_info['id']] = 0
        for ann_info in self.data['annotations']:
            print(cat_stat)
            cat_stat[ann_info['category_id']] += 1

        return cat_stat

    def extract_json(self, target_cat_id, data, cocoapi, img_dir=None, out_img_dir=None):
        '''
        Get images without certain cat
        '''
        import shutil
        cat_dict = {} # imageid -> ann_id for this cat
        for ann_info in data['annotations']:
            ann_id = ann_info['id']
            image_id = ann_info['image_id']
            cat_id = ann_info['category_id']

            if cat_id == target_cat_id:
                if image_id not in cat_dict:
                    cat_dict[image_id] = ann_id

        print(len(cat_dict))
        data_new = data.copy()
        data_new['images'] = []
        data_new['annotations'] = []
        for img_info in data['images']:
            img_id = img_info['id']
            if img_id not in cat_dict:
                if out_img_dir is not None:
                    file = os.path.join(img_dir, img_info['file_name'])
                    out_file = os.path.join(out_img_dir, img_info['file_name'])
                    shutil.copyfile(file, out_file)

                data_new['images'].append(img_info)

                anns = self.get_anns(img_info['file_name'], data, cocoapi)
                data_new['annotations'] += anns
        return data_new

    def remove_repeat_instance(self, target_cat_id, data, cocoapi):
        '''
        Origin can only appear once in one image. This function remove duplicated ones.
        '''
        cat_dict = {} # imageid -> ann_id for this cat
        data_new = data.copy()
        data_new['annotations'] = []
        for ann_info in data['annotations']:
            ann_id = ann_info['id']
            image_id = ann_info['image_id']
            cat_id = ann_info['category_id']

            if cat_id == target_cat_id:
                if image_id not in cat_dict:
                    cat_dict[image_id] = ann_id
                    data_new['annotations'].append(ann_info)
                else:
                    print('repeated insances: ', image_id)
            else:
                data_new['annotations'].append(ann_info)

        return data_new

    def get_defective_imgs(self, data=None, cocoapi=None, img_dir=None, out_img_dir=None):
        if data is None:
            data, cocoapi = self.load_json(self.json_file)

        defect_images = []
        defect_anns = []
        for image_info in data['images']:
            file_name = image_info['file_name']
            img_id = image_info['id']
            ann_id = cocoapi.getAnnIds(img_id)
            annotations = cocoapi.loadAnns(ann_id)
            if len(annotations)> 1:
                defect_images.append(image_info)
                anns_info = self.get_anns(file_name, data, cocoapi)

                defect_anns.append(anns_info)

                if out_img_dir is not None and img_dir is not None:
                    import shutil
                    shutil.copyfile(os.path.join(img_dir, file_name), os.path.join(out_img_dir, file_name))

        data_new = data.copy()
        data_new['images'] = defect_images
        data_new['annotations'] = defect_anns
        return data_new

    def get_defect_free_imgs(self):
        defect_free_imgs = []
        for img_info in self.data['images']:
            anns = self.get_anns(img_info['file_name'])

            if len(anns) == 0:
                defect_free_imgs.append(img_info['file_name'])

        return defect_free_imgs

    def get_anns(self, img_name, json_data=None):
        # get the annotation given the image name
        # get image id for this image

        if json_data is None:
            data = self.data
        else:
            data = json_data

        img_id = ''
        img_id = [img_info['id'] for img_info in data['images'] if img_info['file_name'] == img_name]
        assert len(img_id) == 1, [img_name, img_id]
        img_id = img_id[0]

        if img_id is None:
            print('Cannot find the image')
        # get the annotation using image id
        anns_info = [ann_info for ann_info in data['annotations'] if ann_info['image_id'] == img_id]
        return anns_info

    def remove_non_defective_image(self, out_json_file):
        images_new = []
        for image_info in self.data['images']:
            anns = self.get_anns(image_info['file_name'])
            if len(anns) != 0: 
                images_new.append(image_info)

        new_data = copy.deepcopy(self.data)
        new_data['images'] = images_new

        with open(out_json_file, 'w') as f:
            json.dump(new_data, f)
            #     annotations = coco_api.loadAnns(ann_id)
        #     if len(annotations) > 0:
        #         defect_images.append(file_name)
        # return defect_images

    def change_img_extension(self, old_extension, new_extension, json_file=None):
        # change img extension from old_extension to new_extension
        if json_file is None:
            json_file = self.json_file

        with open(json_file) as j:
            data = json.load(j)

        for image_info in data['images']:
            print(image_info['file_name'])
            image_info['file_name'] = image_info['file_name'].replace(old_extension, new_extension)
            print(image_info['file_name'])
        with open(json_file, 'w') as j:
            json.dump(data, j)

    def change_file_name(self, data=None, save=False):
        '''
        Remove the dir info in the file name
        '''
        if data is None:
            data, _ = self.load_json()
        for image_info in data['images']:
            image_info['file_name'] = os.path.basename(image_info['file_name'])

        if save:
            self.dump_json(data, self.json_file)
        return data

    # crop out encap area
    # need to rewrite image, change annotations for defects, change the height&width for images in json
    def crop_out_encap(self):
        image_dir = self.img_dir
        data = self.data
        data = self.cocoapi
        print('^^^^^^^^^^^^^^ Crop out encap area ^^^^^^^^^^^^^^')
        _, _, cat_name_id_map = self.cat_stat()

        num_images = len(data['images'])
        num_anns = len(data['annotations'])
        print('originally---number of images: ', num_images, 'num of annotations: ', num_anns)

        # crop out encap area for each image
        new_annotations = []
        new_images = []
        for image_info in data['images']:
            # get annotations
            image_id = image_info['id']
            file_name = image_info['file_name']

            ann_ids = cocoapi.getAnnIds(image_id)
            anns = cocoapi.loadAnns(ann_ids)

            assert(len(anns) != 0)

            # get bbox for encap
            for ann_info in anns:
                if ann_info['category_id'] == cat_name_id_map['Encap']: # encap
                    bbox = ann_info['bbox'] # x y w h

            bbox = [int(x) for x in bbox]
            assert(len(bbox) != 0)

            # crop out the encap area
            image = cv2.imread(f'{image_dir}/{file_name}')
            encap_area = image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            # output_dir = image_dir.replace('encap_photos', 'encaps')
            # os.makedirs(output_dir, exist_ok=True)
            # print(output_dir+'/' +os.path.basename(image_file))
            # cv2.imwrite(output_dir+'/' +os.path.basename(image_file), encap_area)

            h, w, _ = encap_area.shape
            image_info['height'] = h
            image_info['width'] = w

            new_images.append(image_info)

            # adjust the location of defect annotations
            for ann_info in anns:
                if ann_info['category_id'] != cat_name_id_map['Encap']: # not encap
                    ann_info['bbox'][0] = round(ann_info['bbox'][0]-bbox[0], 2)
                    ann_info['bbox'][1] = round(ann_info['bbox'][1]-bbox[1], 2)

                    n = len(ann_info['segmentation'])
                    assert(n==1)

                    poly = ann_info['segmentation'][0]
                    new_poly = []
                    n = len(poly)
                    for i in range(0, n, 2):
                        new_poly.append(round(poly[i]-bbox[0], 2))
                        new_poly.append(round(poly[i+1]-bbox[1], 2))

                    ann_info['segmentation'][0] = new_poly

                    new_annotations.append(ann_info)

        print('new num of annotations', len(new_annotations))
        data['annotations'] = new_annotations
        data['images'] = new_images

        return data

    def sanity_check(self):
        cat_stat= self.cat_stat()
        cat_id_name_map, cat_name_id_map = self.get_cat_map()

        # get images with annotations only
        imgs_w_encaps = []
        no_encaps = []
        new_annotations = []
        image_ids = set()
        ann_ids = set()
        for image_info in self.data['images']:
            image_id = image_info['id']
            image_name = image_info['file_name']

            if image_id in image_ids:
                print(image_name, ' img id is repeated')
            image_ids.add(image_id)

            anns = self.get_anns(image_name)
            for ann_info in anns:
                ann_id = ann_info['id']
                if ann_id in ann_ids:
                    print(image_name, ann_id, ' ann id is repeated')
                ann_ids.add(ann_id)

    def create_img_entry(self, img_id, width, height, file_name):
        return {"id": img_id, "width": width, "height": height, \
                "file_name": file_name}
                # "license": 0, "flickr_url": "", "coco_url": "", "date_captured": 0}

    def create_ann_entry(self, img_id, ann_id, category_id, segmentation, area, bbox):
        return {"id": ann_id, "image_id": img_id, "category_id": category_id, \
                "segmentation": segmentation, "area": area, "bbox": bbox}
        #  "iscrowd": 0"attributes": {"not sure": False, "occluded": False}

    # def save_instance(self, img_file, instances, cat_id, box_format='XYWH', out_json_file=None, convertbox=False):
    #     '''
    #     For single image
    #     Convert detectron prediction(box:x1, y1, x2, y2) into coco (box:x1, y1, w, h)
    #     and save the instance to coco json file 
    #     '''
    #     image = cv2.imread(img_file)
    #     img_h, img_w, _ = image.shape

    #     num_instance = len(instances)

    #     # get ann information
    #     boxes = instances.pred_boxes.tensor.numpy()
    #     if box_format == 'XYWH':
    #         boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS) # [x1, y1, x2, y2] to [x1, y1, w, h]
    #     boxes = boxes.tolist()
    #     scores = instances.scores.tolist()
    #     classes = instances.pred_classes.tolist()
    #     has_mask = instances.has("pred_masks")

    #     polygons = []
    #     if has_mask and convertbox is False:
    #         for k in range(num_instance):
    #             # method 1
    #             pred_mask = np.asarray(instances.pred_masks[k])

    #             generic_mask = GenericMask(pred_mask, img_h, img_w)
    #             polygon_det2, _ = generic_mask.mask_to_polygons(pred_mask)
    #             polygon_det2 = np.array(polygon_det2).tolist()
    #             # polygon_xing = mask2poly(mask) # both works
    #             polygons.append(polygon_det2)

    #             # method 2
    #             # for k in range(num_instance):
    #             #     pred_mask = np.asarray(instances.pred_masks[k])
    #             #     polygon = mask2poly(pred_mask)
    #             #     polygons.append(polygon)
    #     else: # convert box to polygon 
    #         for k in range(num_instance):
    #             bbox =  [int(x) for x in boxes[k]] # x y w h
    #             polygon = [[bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], bbox[0], bbox[1]+bbox[3]]]
    #             polygons.append(polygon)

    #     # sample ann entry and image entry
        
    #     _, _, _, ann_entry, img_entry = self.json_info()
    #     out_data = copy.deepcopy(self.data)
    #     out_data['annotations'] = []
    #     out_data['images'] = []
        
    #     # write image entry

    #     img_entry = self.create_img_entry(0, img_w, img_h, os.path.basename(img_file))
    #     out_data['images'].append(img_entry)

    #     # write annotation
    #     current_ann_id = 0
    #     for i in range(num_instance):
    #         ann_entry = self.create_ann_entry(0, current_ann_id, cat_id, polygons[i], 0, boxes[i])
    #         current_ann_id+=1
    #         out_data['annotations'].append(ann_entry)

    #     with open(out_json_file, 'w') as j:
    #         json.dump(out_data, j)

    #     return out_data

    def add_json(self, json_files, exclude_list=None, include_list=None):
        '''
        Add json files to another json file
        Each json file has one image
        '''
        max_ann_id, max_img_id, img_name_id_map, ann_entry, _ = self.json_info()

        for json_file in json_files:
            with open(json_file) as j:
                data2add = json.load(j)
            print('number of images in the json file to add ', len(data2add['images']))

            # change and add image entry
            for img_info in data2add['images']:
                img_name = img_info['file_name']
                target_img_id = self.get_imgid(img_name)
                if target_img_id is None:
                    target_img_id = max_img_id + 1
                    max_img_id += 1
                    img_info['id'] = target_img_id
                    self.data['images'].append(img_info)
                print('target_img_id', target_img_id, 'img_name', img_name)

                # change and add ann entry
                anns_info = self.get_anns(img_name, json_data=data2add)
                print('len of anns---', len(anns_info))
                for ann in anns_info:
                    ann['image_id'] = target_img_id
                    ann['id'] = max_ann_id + 1
                    max_ann_id += 1
                    self.data['annotations'].append(ann)
        return self.data

        # # check if there is duplicate
        # ann_ids = set()
        # for ann in ann_all:
        #     if ann['id'] in ann_ids:
        #         print('duplicate: ', ann['id'])
        #     ann_ids.add(ann['id'])

        # new_data = copy.deepcopy(single_data)
        # new_data['images']= images_all
        # new_data['annotations'] = ann_all

        # num_images = len(new_data['images'])
        # num_anns = len(new_data['annotations'])
        # print('After combined, number of images: ', num_images, 'num of annotations: ', num_anns)

        # self.dump_json(self.data)


    def combine_json(self, json_files, exclude_list=None, include_list=None):
        '''
        Combine several json files
        1. Change image id
        2. Change annotation id
        '''
        images_all = []
        ann_all = []

        annId_curr = 0
        imgId_curr = 0
        for json_file in json_files:
            print('-----------', json_file, '-----------')
            with open(json_file) as j:
                single_data = json.load(j)
            single_cocoapi = COCO(json_file)
            print('number of images', len(single_data['images']))

            for image_info in single_data['images']:
                if exclude_list:
                    if image_info['file_name'] in exclude_list:
                        continue
                if include_list:
                    if image_info['file_name'] not in include_list:
                        continue

                image_id = image_info['id']
                image_info['id'] = imgId_curr
                imgId_curr += 1
                images_all.append(image_info)

                # get the anns
                ann_ids = single_cocoapi.getAnnIds(image_id)
                anns = single_cocoapi.loadAnns(ann_ids)

                for ann in anns:
                    ann['image_id'] = image_info['id']
                    ann['id'] = annId_curr
                    annId_curr += 1
                    ann_all.append(ann)

        # check if there is duplicate
        ann_ids = set()
        for ann in ann_all:
            if ann['id'] in ann_ids:
                print('duplicate: ', ann['id'])
            ann_ids.add(ann['id'])

        new_data = copy.deepcopy(single_data)
        new_data['images']= images_all
        new_data['annotations'] = ann_all

        num_images = len(new_data['images'])
        num_anns = len(new_data['annotations'])
        print('After combined, number of images: ', num_images, 'num of annotations: ', num_anns)

        self.dump_json(new_data)

        return new_data


    def combine_to_json(self, json_file, new_data):
        '''
        Combine several json files
        1. Change image id
        2. Change annotation id
        '''

        # add info of old json file
        if json_file is None:
            json_file = self.json_file
        data_old, _, max_ann_id, max_img_id, img_name_id_map, ann_entry = self.json_info(json_file)

        print('Before combined, number of images: ', len(data_old['images']), ' num of annotations: ', len(data_old['annotations']))

        max_img_id += 1
        max_ann_id += 1
        # add new data
        img_id_map = {}
        for image_info in new_data['images']:
            img_id_map[image_info['id']] = max_img_id
            image_info['id'] = max_img_id
            max_img_id += 1
            data_old['images'].append(image_info)

        for ann_info in new_data['annotations']:
            ann_info['image_id'] = img_id_map[ann_info['image_id']]
            ann_info['id'] = max_ann_id
            max_ann_id += 1
            data_old['annotations'].append(ann_info)

        # check if there is duplicate
        ann_ids = []
        for ann in data_old['annotations']:
            if ann['id'] in ann_ids:
                print('duplicate: ', ann['id'])
            ann_ids.append(ann['id'])

        # combined_data = copy.deepcopy(data_old)
        # combined_data['images']= images_all
        # combined_data['annotations'] = ann_all

        num_images = len(data_old['images'])
        num_anns = len(data_old['annotations'])
        print('After combined, number of images: ', num_images, 'num of annotations: ', num_anns)

        return data_old


    def addPred2json(self, pred_files, cat_name, json_file=None, out_jsonfile=None):
        '''
        pred_files: predictions files storing instances. Assume there is only one cat to add
        '''
        if json_file is None:
            json_file = self.json_file
        print('number of predictions to add: ', len(pred_files))
        data, cocoapi, max_ann_id, img_name_id_map, ann_entry = self.json_info()
        print('max_ann_id', max_ann_id)

        # Change cat
        cat_infos = data['categories']
        cat_names = [c['name'] for c in cat_infos]
        if cat_name not in cat_names: # need to double check
            cat_id = max([cat['id'] for cat in cat_infos]) + 1
            cat_entry = self.create_cat_entry(cat_id, cat_name)
            data['categories'].append(cat_entry)
        else:
            for cat_info in cat_infos:
                if cat_info['name'] == cat_name:
                    cat_id = cat_info['id']
        print('cat_id of ', cat_name, ' is ', cat_id)

        # Combine annotation
        for pred_file in pred_files:
            with open(pred_file) as j:
                prediction = json.load(j)

            if len(np.array(prediction['polygon'])) == 0:
                continue
            polygon = np.array(prediction['polygon'])[0].tolist()
            boxes = prediction['boxes'][0] # detectron2 output format()

            # print((polygon))
            # print(boxes)
            # convert boxes format
            width = boxes[2] - boxes[0]
            height = boxes[3] - boxes[1]
            boxes = [boxes[0], boxes[1], width, height]

            image_name = os.path.basename(pred_file).replace('json', 'jpg')
            image_id = img_name_id_map[image_name]

            ann = ann_entry.copy()
            ann['id'] = max_ann_id+1
            ann['area']=0 # need to change xing
            ann['bbox'] = boxes
            max_ann_id+=1
            ann['category_id'] = cat_id
            ann['image_id'] = image_id
            ann['segmentation']=polygon

            # print('\n')
            # print('ann_entry', ann_entry)
            # print('ann', ann)
            # print('\n')

            data['annotations'].append(ann)

        if out_jsonfile:
            with open(out_jsonfile, 'w') as j:
                json.dump(data, j)

        return data


    def partition_dataset(self, ratio=None, test_imgnames=[]):
        '''
        Divide the json file into train, val, test, according to images. Only guideline is keep def free images according to the provided ratio 
        '''
        if ratio is None:
            tr_ratio, val_ratio, te_ratio = 0.8, 0.1, 0.1
        else:
            tr_ratio, val_ratio, te_ratio = ratio[0], ratio[1], ratio[2]

        # total number of images and ann
        N_images = len(self.data['images'])

        #  get def images and def free images
        def_imgs = []
        def_free_imgs = []
        test_imgs = []
        for img_info in self.data['images']:
            anns = self.get_anns(img_info['file_name'])
            if len(anns) == 0:
                def_free_imgs.append(img_info)
            else:
                if img_info['file_name'] in test_imgnames:
                    test_imgs.append(img_info)
                else:
                    def_imgs.append(img_info)

        N_nodef = len(def_free_imgs)
        N_def = len(def_imgs)
        N_test = len(test_imgs)
        assert N_nodef + N_def + N_test == N_images, 'Numbers of images dont match!'

        random.shuffle(def_free_imgs)
        random.shuffle(def_imgs)

        Ntr_def_free,  Ntr_def  = int(N_nodef * tr_ratio),  int(N_def * tr_ratio)
        Nval_def_free, Nval_def = int(N_nodef * val_ratio), int(N_def * val_ratio)
        Nte_def_free,  Nte_def  = N_nodef - Ntr_def_free - Nval_def_free, N_def - Ntr_def - Nval_def

        te_img_info = def_free_imgs[: Nte_def_free] + def_imgs[: Nte_def] + test_imgs
        val_img_info = def_free_imgs[Nte_def_free : Nte_def_free+Nval_def_free] + def_imgs[Nte_def: Nte_def+Nval_def]
        tr_img_info = def_free_imgs[Nte_def_free+Nval_def_free : ] + def_imgs[Nte_def+Nval_def: ]

        assert(len(tr_img_info) + len(te_img_info) + len(val_img_info)== N_images)

        # get annotations
        tr_anns = []
        te_anns = []
        val_anns = []
        for img_info in tr_img_info:
            ann_ids = self.cocoapi.getAnnIds(img_info['id'])
            anns = self.cocoapi.loadAnns(ann_ids)
            tr_anns += anns
        for img_info in val_img_info:
            ann_ids = self.cocoapi.getAnnIds(img_info['id'])
            anns = self.cocoapi.loadAnns(ann_ids)
            val_anns += anns
        for img_info in te_img_info:
            ann_ids = self.cocoapi.getAnnIds(img_info['id'])
            anns = self.cocoapi.loadAnns(ann_ids)
            te_anns += anns

        # final partition info
        cat_id_name_map, _ = self.get_cat_map()
        cat_sts_tr = {}
        cat_sts_te = {}
        cat_sts_val = {}
        for cat_id in cat_id_name_map:
            cat_sts_tr[cat_id] = 0
            cat_sts_te[cat_id] = 0
            cat_sts_val[cat_id] = 0

        for ann in tr_anns:
            cat_sts_tr[ann['category_id']] += 1
        for ann in te_anns:
            cat_sts_te[ann['category_id']] += 1
        for ann in val_anns:
            cat_sts_val[ann['category_id']] += 1

        print('----------- Categories statistics ----------')
        for cat_key in cat_sts_tr:
            length = len(cat_id_name_map[cat_key][:6]) + 4
            print(str(cat_id_name_map[cat_key][:6]).ljust(length), end='')
        print('\ntrain')
        for cat_key in cat_sts_tr:
            length = len(cat_id_name_map[cat_key][:6]) + 4
            print(str(cat_sts_tr[cat_key]).ljust(length), end='')
        print('\ntest')
        for cat_key in cat_sts_te:
            length = len(cat_id_name_map[cat_key][:6]) + 4
            print(str(cat_sts_te[cat_key]).ljust(length), end='')
        print('\nvalidation')
        for cat_key in cat_sts_val:
            length = len(cat_id_name_map[cat_key][:6]) + 4
            print(str(cat_sts_val[cat_key]).ljust(length), end='')
        print('\n')

        print('number of training images: ', len(tr_img_info))
        print('number of test images: ', len(te_img_info))
        print('number of val images: ', len(val_img_info))
        print('number of training annotations: ', len(tr_anns))
        print('number of test annotations: ', len(te_anns))
        print('number of val annotations: ', len(val_anns))

        print('total annotation ratio: tr/te, tr/val ', len(tr_anns)/len(te_anns), len(tr_anns)/len(val_anns))

        train_data = copy.deepcopy(self.data)
        train_data['images'] = tr_img_info
        train_data['annotations'] = tr_anns
        test_data = copy.deepcopy(self.data)
        test_data['images'] = te_img_info
        test_data['annotations'] = te_anns
        val_data = copy.deepcopy(self.data)
        val_data['images'] = val_img_info
        val_data['annotations'] = val_anns

        return train_data, test_data, val_data

    def get_info(self, target_defect, img_dir=''):
        cat_id_name_map, cat_name_id_map = self.get_cat_map()
        # target_defect: cat name
        infos = {}
        for image_info in self.data['images']:
            anns_info = self.get_anns(image_info['file_name'])

            target_defect_info = []
            for ann_info in anns_info:
                if cat_id_name_map[ann_info['category_id']] == target_defect:
                    target_defect_info.append(ann_info)
                    
            if len(target_defect_info) > 0:
                key = image_info['file_name'] if img_dir==None else os.path.join(img_dir, image_info['file_name'])
                infos[key] = target_defect_info
        return infos

