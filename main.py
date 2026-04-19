import cv2
from ultralytics import YOLO

#Check the yaml file: ReID was enabled. Docs: https://docs.ultralytics.com/modes/track/

# using yolo v26 model (latest model) nano version which is most efficient 
model = YOLO('yolo26n.pt')

#model.track tracks objects and assigns each one a unique ID (good for student recognition)
results = model.track(source='0', tracker = 'config.yaml', show=True, classes=[0])




  
    
     