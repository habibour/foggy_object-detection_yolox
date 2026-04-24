import os
import xml.etree.ElementTree as ET
import random
import shutil

# Configuration
classes = ["person", "bicycle", "car", "bus", "motorbike"] 
input_dir = "RTTS" # Ensure your RTTS folder is in the same directory
output_dir = "RTTS_Ready"
# Split targets from the paper: 2592 Train, 865 Val, 865 Test 
split_counts = {'train': 2592, 'val': 865, 'test': 865}

# Create output folders
for split in split_counts.keys():
    os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

def convert_coords(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def convert_xml(xml_path, label_output_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(label_output_path, 'w') as f:
        for obj in root.iter('object'):
            cls_name = obj.find('name').text.lower()
            if cls_name not in classes:
                continue
            cls_id = classes.index(cls_name)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert_coords((w, h), b)
            f.write(f"{cls_id} {' '.join([f'{a:.6f}' for a in bb])}\n")

# Get file list and shuffle
all_files = [f[:-4] for f in os.listdir(os.path.join(input_dir, 'Annotations')) if f.endswith('.xml')]
random.seed(42)
random.shuffle(all_files)

# Partition and move
cursor = 0
for split, count in split_counts.items():
    subset = all_files[cursor : cursor + count]
    cursor += count
    
    for filename in subset:
        # Move Image (.png based on file list)
        src_img = os.path.join(input_dir, 'JPEGImages', f"{filename}.png")
        dst_img = os.path.join(output_dir, 'images', split, f"{filename}.png")
        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)
            
        # Convert and Move Label
        src_xml = os.path.join(input_dir, 'Annotations', f"{filename}.xml")
        dst_label = os.path.join(output_dir, 'labels', split, f"{filename}.txt")
        if os.path.exists(src_xml):
            convert_xml(src_xml, dst_label)

print(f"Preprocessing finished. Data ready in {output_dir}")
