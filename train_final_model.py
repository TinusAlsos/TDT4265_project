from utils import convert_RDD2022_to_darknet_format, create_empty_txts
import shutil
import os
ROOT = os.path.dirname(os.path.abspath(__file__))

def main():

    # Setup the folder stucture and preprocess the data
    os.system('jupyter nbconvert --execute --to notebook --inplace data_set_prep.ipynb')

    convert_RDD2022_to_darknet_format(mod = True)
    os.chdir(os.path.join(ROOT, 'yolov5'))
    train_final_model = 'python train.py --epochs 50 --batch-size -1 --img 960 --data all_data2.yaml --device 0 --hyp data/hyps/norway2 --weights runs/train/Norway_model/weights/last.pt --name final_model  --cache'
    detect_final_model = 'python detect.py --img 1920 --conf-thres 0.00001 -iou-thres 0.85 --weights runs/train/final_model/weights/best.pt --source ../RDD2022/Norway/test/images --name final_model --save-txt --save-conf'
    os.system(train_final_model)
    os.system(detect_final_model)
    os.chdir(ROOT)

    predicts_file = 'runs/detect/final_model'

    predicts_file_path = 'yolov5/' + predicts_file
    predicts_label_path =predicts_file_path + '/labels'

    create_empty_txts(predicts_file_path)

    shutil.make_archive('send_to_your_computer', format='zip', root_dir=predicts_label_path)


    os.system('python submit.py')
    print('The final submission file should be found in /submissions and is output and some number. The highest number is the correct output file.')



if __name__ == '__main__':
    main()