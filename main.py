import cv2
from ultralytics import YOLO
import torchreid
import torch
import numpy
from torchreid.reid.utils import feature_extractor
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image


frames_elapsed = 0
students = {}
#Temporal smoothing - this "averages" appearance over n frames
buffer = 30

#Temp tracks - these are the boxes detected by model (but not added to students)
temp_tracks = {}

#Temp frames - used to check if the track has accumulated enough frames (buffer) to add to students
temp_frames = {}

device = "cpu"

#How much appearance matters in matching
appearence_weight = .6

#How much face matters in matching
face_weight = .4


# using yolo v26 model (latest model) nano version which is most efficient
model = YOLO('yolo26n.pt')

#Resnet for generating face embeddings

resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

#Osnet extracts appearence embeddings (clothes, body shape, colors etc)
image_feature_extractor = feature_extractor.FeatureExtractor(
    model_name = "osnet_x0_75",
    model_path = "ITCS-4152-CV-Project/osnet_x0_75_imagenet.pth",
    device = "cpu"
)

#Adjust as needed
config_path = r""

#finds faces in an image and crops it
mtcnn = MTCNN(image_size=160, margin=20, device=device)

#model.track tracks objects and assigns each one a unique ID 
for r in model.track(source=r'ITCS-4152-CV-Project\IMG_7593.mp4', tracker=config_path, show=True, classes=[0], stream=True):
    

    #the image to work with
    frame = r.orig_img
    print(str(len(students)) + str(" Unique students being tracked"))

    # n buffer frames have past - reset all temporary tracks
    if frames_elapsed > 1 and frames_elapsed % buffer == 0:
        
        temp_frames = {} 
        temp_tracks = {}
    frames_elapsed += 1

    #For each box yolo detects
    for box in r.boxes:
        #set box unique by default 
        unique = True

        #Crop box from whole image
        if box and box.id and box.id.item():
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        #Extract the features from the box with OSnet- captures appearence
        new_feature_vector = image_feature_extractor(crop)
        new_feature_vector = new_feature_vector.squeeze(0)

        #Crop the face from the cropped box
        face_tensor = mtcnn(crop)
        if face_tensor is None:
            #If the face cant be recognized, initialize with 0's  *****SHOULD IMPROVE THIS*************
            face_embedding = torch.zeros(512)
            
        else:
            with torch.no_grad():
                #Otherwise, generate the face embedding
                face_embedding = resnet(face_tensor.unsqueeze(0))

        #Combine the OSnet appearence embedding with the face embedding - now weve encoded appearance + face into one embedding
        face_embedding = face_embedding.squeeze(0)
        hybrid_embedding = torch.cat((new_feature_vector * appearence_weight, face_embedding * face_weight), dim=0).flatten()
        hybrid_embedding = torch.nn.functional.normalize(hybrid_embedding, dim=0)

        #Check existing students - if similar enough, we dont need to do anything
        for student in students.values():
            sim_score = float(torch.cosine_similarity(student, hybrid_embedding, dim=0).item())
            
            #Higher = more lenient with whats considered a "unique" student
            if sim_score > .9:
                
                #Cancels the next loop
                unique = False
                break

        #Didnt match any of existing students- add new person embedding to students 
        if unique:
            id = int(box.id.item())
            if temp_tracks.__contains__(id):
                temp_tracks[id] += hybrid_embedding
                temp_frames[id] += 1
                if temp_frames[id] >= buffer:
                    students[id] = temp_tracks[id] / temp_frames[id]
            else:
                temp_tracks[id] = hybrid_embedding
                temp_frames[id] = 1
