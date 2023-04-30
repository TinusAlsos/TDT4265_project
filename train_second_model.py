from utils import convert_RDD2022_to_darknet_format
import os
import shutil
from utils import create_empty_txts
ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    convert_RDD2022_to_darknet_format('Norway')
    os.chdir(os.path.join(ROOT, 'yolov5'))
    train_second_model = 'python train.py --epochs 40 --img 960 --data Norway.yaml --device 0 --weights runs/train/baseline_model/weights/last.pt --name Norway_model  --cache'
    detect_second_model = 'python detect.py --weights runs/train/Norway_model/weights/best.pt --source ../RDD2022/Norway/test/images --name Norway_model --save-txt --save-conf'
    os.system(train_second_model)
    os.system(detect_second_model)
    os.chdir(ROOT)
    
    

    predicts_file = 'runs/detect/Norway_model'

    predicts_file_path = 'yolov5/' + predicts_file
    predicts_label_path =predicts_file_path + '/labels'

    create_empty_txts(predicts_file_path)

    shutil.make_archive('send_to_your_computer', format='zip', root_dir=predicts_label_path)

    os.system('python submit.py')
    print('The final submission file for the Norway(second) model should be found in /submissions and is output and some number. The highest number is the correct output file.')



if __name__ == '__main__':
    main()