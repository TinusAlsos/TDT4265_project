import os
print(os.getcwd())


from codecarbon import EmissionsTracker

tracker = EmissionsTracker(
    project_name = 'fredag',
    save_to_file=True,
    output_dir='carbon',
    output_file='carbon_fredag'
)

tracker.start()

train1 = 'python train.py --img 960 --batch-size -1  --epochs 45 --device 0 --data all_data2.yaml --hyp data/hyps/fredag.yaml --weights yolov5m6.pt --name lordag'
detect1 = 'python detect.py --img 960 --weights runs/train/lordag/weights/best.pt --source ../RDD2022/Norway/test/images --name fredag_vanlig --save-txt --save-conf'
detect2 = 'python detect.py --img 960 --weights runs/train/lordag/weights/best.pt --source ../RDD2022/Norway/test/images --name fredag_conf001 --conf-thres 0.01 --save-txt --save-conf'
detect3 = 'python detect.py --img 960 --weights runs/train/lordag/weights/best.pt --source ../RDD2022/Norway/test/images --name fredag_conf001 --conf-thres 0.01 --iou-thres 0.7 --save-txt --save-conf'


os.system(train1)
# os.system(detect1)
# os.system(detect2)

# os.system(detect3)

tracker.stop()
