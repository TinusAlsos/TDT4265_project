import os
import xml.dom.minidom
import shutil
import random

ROOT =  os.path.dirname(os.path.abspath(__file__))
RDD2022_DATA_FOLDER = os.path.join(ROOT, 'RDD2022')
label_to_index_map = {'D00': 0, 'D10': 1, 'D20': 2, 'D40': 3}
index_to_label_map = {0: 'D00', 1: 'D10', 2: 'D20', 3: 'D40'}

def pascal_voc_to_yolo_coordinates(xmin: float, ymin: float, xmax: float, ymax: float):
    # Function to convert Pascal VOC coordinates to YOLO coordinates
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin
    return x_center, y_center, w, h

def yolo_to_pascal_voc_coordinates(x_center: float, y_center: float, w: float, h: float):
    # Function to convert YOLO coordinates to Pascal VOC coordinates
    xmin = x_center - w / 2
    ymin = y_center - h / 2
    xmax = x_center + w / 2
    ymax = y_center + h / 2
    return xmin, ymin, xmax, ymax

def noramlize_yolo_format(x_center: float, y_center: float, w: float, h: float, image_width: float, image_height: float):
    # Function to normalize YOLO format coordinates
    x_center /= image_width
    y_center /= image_height
    w /= image_width
    h /= image_height
    return x_center, y_center, w, h

def convert_RDD2022_to_darknet_format(data_set_name: str = 'all_data', num_images_to_convert: int = -1) -> None:
    """ Function to convert RDD2022 dataset to Darknet format
     
    Args:
        data_set_name (str): Name of the dataset to convert
        
    Returns:
        None"""

    image_folder_path = os.path.join(RDD2022_DATA_FOLDER, data_set_name, 'train', 'images')
    annotiation_folder_path = os.path.join(RDD2022_DATA_FOLDER, data_set_name, 'train', 'annotations', 'xmls')
    image_save_folder_path = os.path.join(ROOT, 'datasets', data_set_name, 'images')
    label_save_folder_path = os.path.join(ROOT, 'datasets', data_set_name, 'labels')
    
    if not os.path.exists(image_save_folder_path):
        os.makedirs(image_save_folder_path)
    if not os.path.exists(label_save_folder_path):
        os.makedirs(label_save_folder_path)
    if not os.path.exists(os.path.join(image_save_folder_path, 'train')):
        os.makedirs(os.path.join(image_save_folder_path, 'train'))
    if not os.path.exists(os.path.join(image_save_folder_path, 'val')):
        os.makedirs(os.path.join(image_save_folder_path, 'val'))
    if not os.path.exists(os.path.join(label_save_folder_path, 'train')):
        os.makedirs(os.path.join(label_save_folder_path, 'train'))
    if not os.path.exists(os.path.join(label_save_folder_path, 'val')):
        os.makedirs(os.path.join(label_save_folder_path, 'val'))
    # Get the list of image files
    if data_set_name == 'all_data':
        image_folder_paths = []
        annotiation_folder_paths = []
        for folder_name in os.listdir(RDD2022_DATA_FOLDER):
            image_folder_paths.append(os.path.join(RDD2022_DATA_FOLDER, folder_name, 'train', 'images'))
            annotiation_folder_paths.append(os.path.join(RDD2022_DATA_FOLDER, folder_name, 'train', 'annotations', 'xmls'))
    else:
        image_folder_paths = []
        image_folder_paths.append(image_folder_path)
        annotiation_folder_paths = []
        annotiation_folder_paths.append(annotiation_folder_path)
    for idx, image_folder_path in enumerate(image_folder_paths):
        annotiation_folder_path = annotiation_folder_paths[idx]
        image_files = os.listdir(image_folder_path)
        if num_images_to_convert != -1:
            image_files = image_files[:num_images_to_convert]
        for image_file in image_files:
            if random.randint(0,4) <= 3:
                image_save_path = os.path.join(image_save_folder_path, 'train', image_file)
                label_save_path = os.path.join(label_save_folder_path, 'train', os.path.splitext(image_file)[0] + '.txt')
            else:
                image_save_path = os.path.join(image_save_folder_path, 'val', image_file)
                label_save_path = os.path.join(label_save_folder_path, 'val', os.path.splitext(image_file)[0] + '.txt')
                
            # Copy the image file to the destination folder
            shutil.copy(os.path.join(image_folder_path, image_file), image_save_path)

            # Extract the filename without extension
            filename, extension = os.path.splitext(image_file)
            # Check if corresponding XML annotation file exists
            xml_file = os.path.join(annotiation_folder_path, filename + '.xml')
            if not os.path.exists(xml_file):
                print(f"Annotation file not found for image '{image_file}', skipping...")
                continue
            # Parse the XML annotation file
            dom = xml.dom.minidom.parse(xml_file)
            root = dom.documentElement
            objects = dom.getElementsByTagName("object")
            # Get the image size
            width = float(root.getElementsByTagName('width')[0].childNodes[0].data)
            height = float(root.getElementsByTagName('height')[0].childNodes[0].data)
            # Loop through each object in the annotation
            with open(label_save_path, 'w') as f:
                for obj in objects:
                    xmin = float(obj.getElementsByTagName('xmin')[0].childNodes[0].data)
                    ymin = float(obj.getElementsByTagName('ymin')[0].childNodes[0].data)
                    xmax = float(obj.getElementsByTagName('xmax')[0].childNodes[0].data)
                    ymax = float(obj.getElementsByTagName('ymax')[0].childNodes[0].data)
                    label = obj.getElementsByTagName('name')[0].childNodes[0].data
                    if label not in ['D00', 'D10', 'D20', 'D40']:
                        continue
                    # Convert the coordinates to Darknet format
                    x, y, w, h = pascal_voc_to_yolo_coordinates(xmin, ymin, xmax, ymax)
                    x, y, w, h = noramlize_yolo_format(x, y, w, h, width, height)
                    # Save the Darknet format annotation in a text file
                    darknet_annotation = f"{label_to_index_map[label]} {x} {y} {w} {h}\n"
                    f.write(darknet_annotation)


def create_empty_txts(images_path: str) -> None:
    """ Function to create empty txt files for images in a folder
     
    Args:
        images_path (str): Path to the folder containing images
        
    Returns:
        None"""
    image_files = os.listdir(images_path)
    for image_file in image_files:
        filename, extension = os.path.splitext(image_file)
        if extension == '.jpg':
            txt_file = os.path.join(images_path, 'labels', filename + '.txt')
            if not os.path.exists(txt_file):
                with open(txt_file, 'w') as f:
                    pass

def remove_files_from_folder_by_type(folder_path: str, ending: str) -> None:
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Loop through the files and remove those that end with ".txt"
    for file in files:
        if file.endswith(ending):
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)
            print(f"File '{file}' removed successfully.")

def main():
    data_set_name = 'China_MotorBike'
    convert_RDD2022_to_darknet_format(data_set_name)
    # images_path = os.path.join(ROOT, 'yolov5', 'runs', 'detect', 'exp4')
    # images_path = 'RDD2022/Norway/test/images'
    # create_empty_txts(images_path)
    # remove_files_from_folder_by_type(images_path, '.txt')
    # data_set_name = 'Japan'
    # convert_RDD2022_to_darknet_format(data_set_name)

if __name__ == '__main__':
    main()