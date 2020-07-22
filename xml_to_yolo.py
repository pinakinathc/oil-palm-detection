# -*- coding; utf-8 -*-
# author: www.pinakinathc.me

import argparse
import os
import glob
import xml.etree.ElementTree as ET

def main(xml_filepath, output_dir):
    xml_filename = os.path.split(xml_filepath)[-1]
    yolo_gt = ""

    tree = ET.parse(xml_filepath)
    root = tree.getroot()
    print (root)
    size = root.find('size')
    W = int(size.find('width').text)
    H = int(size.find('height').text)

    for object_ in root.findall('object'):
        bndbox = object_.find('bndbox')
        xmin = int(bndbox.find('xmin').text)/float(W)
        ymin = int(bndbox.find('ymin').text)/float(H)
        xmax = int(bndbox.find('xmax').text)/float(W)
        ymax = int(bndbox.find('ymax').text)/float(H)

        x_centre, y_centre = (xmin+xmax)/2., (ymin+ymax)/2.
        width, height = xmax-xmin, ymax-ymin

        gt_box = list(map(str, [0, x_centre, y_centre, width, height]))
        gt_box = ' '.join(gt_box)+'\n'
        yolo_gt += gt_box

    print (yolo_gt)
    with open(os.path.join(output_dir, xml_filename[:-3]+"txt"), "w") as fp:
        fp.write(yolo_gt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts XML to yolo format")
    parser.add_argument("xml_path", type=str, help="enter root dir of xml path")
    parser.add_argument("output_path", type=str, help="enter root dir to save")
    args = parser.parse_args()
    
    xml_list = glob.glob(os.path.join(args.xml_path, "*.xml"))
    for xml_filepath in xml_list:
        main(xml_filepath, args.output_path)
