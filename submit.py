import json
import zipfile
import os

""" KJØR FRA TDT4265_project """

path_to_zip_file = 'send_to_your_computer.zip'

path_to = 'submissions/predictions'

identifier = str(int(os.listdir(path_to)[-1][-1] + 1))

foldername = 'pred' + identifier

savefolder = os.path.join(path_to, foldername)

with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall(savefolder)

os.remove(path_to_zip_file)

labels_path = os.path.join(savefolder, 'labels')
image_folder_path = 'RDD2022/Norway/test/images'

submission_json_path = f'{path_to}/submission{identifier}.json'
output_json_path = f'{path_to}/output{identifier}.json'

os.system(f'globox convert {labels_path} {submission_json_path} --format yolov5 --save_fmt coco --img_folder {image_folder_path} --coco_auto_ids')

with open(submission_json_path, 'r') as f:
   data = json.load(f)
   anns = data['annotations']
with open(output_json_path,'w') as f:
   json.dump(anns,f)