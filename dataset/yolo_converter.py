import xml.etree.ElementTree as ET
import os # for os.path.basename

def xml_to_yolo(xml_file, classes):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    yolo_format = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        class_id = classes.index(class_name)

        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        width = xmax - xmin
        height = ymax - ymin
        center_x = xmin + width / 2
        center_y = ymin + height / 2

        # Normalize coordinates
        width /= float(root.find('size/width').text)
        height /= float(root.find('size/height').text)
        center_x /= float(root.find('size/width').text)
        center_y /= float(root.find('size/height').text)

        yolo_format.append(f"{class_id} {center_x} {center_y} {width} {height}")

    return yolo_format


all_images_dir = "simple_images/all_images"

classes = []

# find all xml files in that directory and convert them to yolo format
for xml_file in os.listdir(all_images_dir):
    if xml_file.endswith('.xml'):
        # find object tags first and find name tag inside them and get its text
        object_tags = ET.parse(os.path.join(all_images_dir, xml_file)).findall('object')
        for obj in object_tags:
            classes.append(obj.find('name').text)

        classes = list(set(classes)) # remove duplicates

        yolo_format = xml_to_yolo(os.path.join(all_images_dir, xml_file), classes)

        # save yolo format to a text file
        with open(os.path.join(all_images_dir, os.path.splitext(xml_file)[0] + ".txt"), "w") as f:
            f.write("\n".join(yolo_format))


# save classes to a text file
with open(os.path.join(all_images_dir, "classes.txt"), "w") as f:
    f.write("\n".join(classes))

print("Done!")







