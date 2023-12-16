import os

labels_train = "/Users/canrollas/Projects/OCR/dataset/labels/train"
labels_val = "/Users/canrollas/Projects/OCR/dataset/labels/val"


def remove_xml_files():
    for file in os.listdir(labels_train):
        if file.endswith(".xml"):
            os.remove(os.path.join(labels_train, file))
    for file in os.listdir(labels_val):
        if file.endswith(".xml"):
            os.remove(os.path.join(labels_val, file))

remove_xml_files()
