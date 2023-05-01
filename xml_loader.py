# Directory to the xml files
root_dir = '/data/Projects/accessmath-icfhr2018/AccessMathVOC'
train_lectures = ['NM_lecture_01']

for lecture in train_lectures:
    # Directory to the image files
    img_dir = os.path.join(root_dir, lecture, 'JPEGImages')
    xml_dir = os.path.join(root_dir, lecture, 'Annotations')

    # Output jason file path
    output_jsonpath = os.path.join(root_dir, lecture, 'annotation.json')

    xml2coco(xml_dir, output_jsonpath, img_dir)