import os
print(os.getcwd())

train1 = 'python train.py --img 960 --epochs 50 --device 0 --data Norway.yaml --hyp data/hyps/norway.yaml --weights runs/train/natt_til_onsdag_siste_del/weights/best.pt --name natt_til_torsdag_x'
detect1 = 'python detect.py --img 960 --weights runs/train/natt_til_torsdag_x/weights/best.pt --source ../RDD2022/Norway/test/images --name natt_til_torsdag_x --save-txt --save-conf'


train2 = 'python train.py --img 640 --epochs 50 --device 0 --data Norway.yaml --hyp data/hyps/norway.yaml --weights runs/train/natt_til_onsdag_siste_del/weights/best.pt --name natt_til_torsdag_y'
detect2 = 'python detect.py --img 640 --weights runs/train/natt_til_torsdag_y/weights/best.pt --source ../RDD2022/Norway/test/images --name natt_til_torsdag_y --save-txt --save-conf'

os.system(train1)
os.system(detect1)
os.system(train2)
os.system(detect2)
