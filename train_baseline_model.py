from utils import convert_RDD2022_to_darknet_format, create_empty_txts
import shutil
import os
ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    convert_RDD2022_to_darknet_format()
    os.chdir(os.path.join(ROOT, 'yolov5'))
    train_baseline_model = 'python train.py --epochs 110 --data all_data.yaml --device 0 --weights yolov5s.pt --name baseline_model --cache'
    detect_baseline_model = 'python detect.py --weights runs/train/baseline_model/weights/best.pt --source ../RDD2022/Norway/test/images --name baseline_model --save-txt --save-conf'
    os.system(train_baseline_model)
    os.system(detect_baseline_model)
    os.chdir(ROOT)
    
    predicts_file = 'runs/detect/baseline_model'

    predicts_file_path = 'yolov5/' + predicts_file
    predicts_label_path =predicts_file_path + '/labels'

    create_empty_txts(predicts_file_path)

    shutil.make_archive('send_to_your_computer', format='zip', root_dir=predicts_label_path)
    
    os.system('python submit.py')
    print('The final submission file for the baseline model should be found in /submissions and is output and some number. The highest number is the correct output file.')



if __name__ == '__main__':
    main()