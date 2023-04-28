import os


train1 = 'python train.py --img 1600 --epochs 120 --device 0 --data Norway.yaml --batch-size 8 --hyp data/hyps/norway.yaml --weights runs/train/torsdag_y/weights/last.pt --name natt_til_fredag'
detect1 = 'python detect.py --img 1600 --weights runs/train/natt_til_fredag/weights/last.pt --source ../RDD2022/Norway/test/images --name torsdag_y --save-txt --save-conf'


detect_conf = 'python detect.py --img 960 --weights runs/train/torsdag_y/weights/last.pt --source ../RDD2022/Norway/test/images --name torsdag_conf01 --conf-thres 0.1 --save-txt --save-conf'
detect_conf2 = 'python detect.py --img 960 --weights runs/train/torsdag_y/weights/last.pt --source ../RDD2022/Norway/test/images --name torsdag_conf001 --conf-thres 0.01 --save-txt --save-conf'
detect_conf3 = 'python detect.py --img 960 --weights runs/train/torsdag_y/weights/last.pt --source ../RDD2022/Norway/test/images --name torsdag_conf01iou03 --conf-thres 0.1 --iou-thres 0.3 --save-txt --save-conf'


detect_conf2 = 'python detect.py --img 960 --weights runs/train/torsdag_y/weights/last.pt --source ../RDD2022/Norway/test/images --name torsdag_conf0001 --conf-thres 0.001 --save-txt --save-conf'
detect_conf3 = 'python detect.py --img 960 --weights runs/train/torsdag_y/weights/last.pt --source ../RDD2022/Norway/test/images --name torsdag_conf01iou97 --conf-thres 0.1 --iou-thres 0.7 --save-txt --save-conf'


os.system(detect_conf2)
os.system(detect_conf3)
