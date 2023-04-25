import os
import shutil
from utils import create_empty_txts

predicts_file = 'runs/detect/enkelttest3'

predicts_file_path = 'yolov5/' + predicts_file
predicts_label_path =predicts_file_path + '/labels'

create_empty_txts(predicts_file_path)

shutil.make_archive('send_to_your_computer', format='zip', root_dir=predicts_file_path)
