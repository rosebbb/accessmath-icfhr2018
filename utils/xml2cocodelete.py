import xml.etree.ElementTree as ET
import os
import glob
import cv2
import numpy as np
import json
def create_image_info(img_name, img_id, img):
    height, width, _ = img.shape

    image_info = {
        'file_name': img_name,
        'height': height,
        'width': width,
        'id': img_id
    }
    return image_info

def create_ann_info(bbox, cat_id, img_id, ann_id): #[xmin, ymin, xmax, ymax]
    xmin = int(bbox[0])
    ymin = int(bbox[1])
    xmax = int(bbox[2])
    ymax = int(bbox[3])
    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann_info = {
        'area': o_width * o_height,
        'segmentation': [[xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]], 
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': cat_id,
        'image_id': img_id, 
        'id': ann_id,
        'iscrowd': 0

    }

    return ann_info

def load_object_xml(root):
    name = root.find('name').text
    bndbox = root.find('bndbox')

    xmin = float(bndbox.find("xmin").text)
    ymin = float(bndbox.find("ymin").text)
    xmax = float(bndbox.find("xmax").text)
    ymax = float(bndbox.find("ymax").text)

    return name, xmin, ymin, xmax, ymax



def get_ann(img, input_xml_file):
    h, w = img.shape[0:2]

    tree = ET.parse(input_xml_file)
    root = tree.getroot()
    # .text
    filename = root.find('filename').text
    objects = root.findall("object")
    bboxes = []
    words = []
    for o in objects:
        name, xmin, ymin, xmax, ymax = load_object_xml(o)
        if name == 'text':
            bbox = [xmin, ymin, xmax, ymax]
            bboxes.append(bbox)
            words.append('???')
    return np.array(bboxes), words

def xml2coco(ann_dir, output_jsonpath, img_dir):
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    output_json_dict['categories'].append({'supercategory': 'none', 'id': 1, 'name': 'text'})
    ann_id = 0
    img_id = 0
    print('Start converting !')

    ann_files_wo_img = []
    img_files_wo_ann = []
    for image_file in glob.glob(os.path.join(img_dir, '*.png')):
        print(img_id, image_file)
        image_name = os.path.basename(image_file)
    # for ann_file in glob.glob(os.path.join(ann_dir, '*.xml')):
        ann_file = os.path.join(ann_dir, image_name.replace('png', 'xml'))
        # image_file = os.path.join(os.path.join(img_dir, os.path.basename(ann_file).replace('xml', 'jpg')))
        if not os.path.isfile(image_file):
            print('img file not found!')
            ann_files_wo_img.append(os.path.basename(ann_file))
            continue

        if not os.path.isfile(ann_file):
            print('ann file not found!')
            img_files_wo_ann.append(os.path.basename(image_file))
            continue

        img = cv2.imread(image_file)
        bboxes, _ = get_ann(img, ann_file) #[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
        img_info = create_image_info(image_name, img_id, img)
        output_json_dict['images'].append(img_info)

        for bbox in bboxes:
            ann_info = create_ann_info(bbox, 1, img_id, ann_id)
            output_json_dict['annotations'].append(ann_info)
            ann_id = ann_id + 1

        img_id += 1

    with open(output_jsonpath, 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)

    print(len(ann_files_wo_img))

    print(len(img_files_wo_ann))


# Directory to the xml files
root_dir = '/data/Projects/accessmath-icfhr2018/AccessMathVOC'
train_lectures = ['lecture_06','lecture_18']

for lecture in train_lectures:
    # Directory to the image files
    img_dir = os.path.join(root_dir, lecture, 'JPEGImages')
    xml_dir = os.path.join(root_dir, lecture, 'Annotations')

    # Output jason file path
    output_jsonpath = os.path.join(root_dir, lecture, 'annotation.json')

    xml2coco(xml_dir, output_jsonpath, img_dir)