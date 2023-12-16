import os
import shutil

test_images_names = os.listdir("testimages")
train_images_names = os.listdir("trainimages")

list_of_test_images = []  # jpg files
list_of_txt_test_files = []  # txt files
list_of_train_images = []  # jpg files
list_of_txt_train_files = []  # txt files
list_of_xml_train_files = []  # xml files
list_of_xml_test_files = []  # xml files

for image_name in test_images_names:
    if image_name.endswith(".jpg"):
        # append absolute path to the image name
        list_of_test_images.append(os.path.join("testimages", image_name))

    elif image_name.endswith(".txt"):
        # append absolute path to the text file name
        list_of_txt_test_files.append(os.path.join("testimages", image_name))

    elif image_name.endswith(".xml"):
        # append absolute path to the xml file name
        list_of_xml_test_files.append(os.path.join("testimages", image_name))

for image_name in train_images_names:
    if image_name.endswith(".jpg"):
        # append absolute path to the image name
        list_of_train_images.append(os.path.join("trainimages", image_name))

    elif image_name.endswith(".txt"):
        # append absolute path to the text file name
        list_of_txt_train_files.append(os.path.join("trainimages", image_name))

    elif image_name.endswith(".xml"):
        # append absolute path to the xml file name
        list_of_xml_train_files.append(os.path.join("trainimages", image_name))

print("Number of test images:", len(list_of_test_images))
print("Number of test text files:", len(list_of_txt_test_files))
print("Number of train images:", len(list_of_train_images))
print("Number of train text files:", len(list_of_txt_train_files))
print("Number of train xml files:", len(list_of_xml_train_files))
print("*****" * 40)

images = list_of_test_images + list_of_train_images
txt_files = list_of_txt_test_files + list_of_txt_train_files
xml_files = list_of_xml_test_files + list_of_xml_train_files

all_images = []


def find_xml_from_jpg(jpg_file):
    splitted_name = jpg_file.split("/")[-1].split(".")[0]
    print(splitted_name)
    for xml_file in xml_files:
        print(xml_file, splitted_name)
        if splitted_name in xml_file:
            return xml_file


def find_txt_from_jpg(jpg_file):
    splitted_name = jpg_file.split("/")[-1].split(".")[0]
    for txt_file in txt_files:
        if splitted_name in txt_file:
            return txt_file


for image in images:
    xml_file = find_xml_from_jpg(image)
    txt_file = find_txt_from_jpg(image)
    all_images.append((image, xml_file, txt_file))

yolo_dir = "../dataset"

# Move files to YOLO directory structure
for image_path, xml_path, txt_path in all_images:
    # Determine whether it belongs to the training or validation set based on the YOLO directory structure
    dest_dir = 'train' if 'train' in image_path else 'val'

    # Move image
    shutil.move(image_path, os.path.join(yolo_dir, 'images', dest_dir, os.path.basename(image_path)))

    # Move XML file (if available)
    if xml_path:
        shutil.move(xml_path, os.path.join(yolo_dir, 'labels', dest_dir, os.path.basename(xml_path)))

    # Move text file (if available)
    if txt_path:
        shutil.move(txt_path, os.path.join(yolo_dir, 'labels', dest_dir, os.path.basename(txt_path)))
