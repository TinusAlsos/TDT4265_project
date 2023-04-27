import os
train1 = 'python train.py --img 960 --epochs 50 --device 0 --data Norway.yaml --hyp data/hyps/norway2.yaml --weights runs/train/natt_til_onsdag_siste_del/weights/best.pt --name torsdag_x'
detect1 = 'python detect.py --img 960 --weights runs/train/torsdag_x/weights/best.pt --source ../RDD2022/Norway/test/images --name torsdag_x --save-txt --save-conf'

os.system(train1)
os.system(detect1)