import cv2
# import socket
# import pickle
# import os
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
resnet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)
embedding_list, name_list = torch.load('data.pt')

# s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1000000)
# server_ip = "0.0.0.0"
# server_port = 8000


def face_match(img_path):
    img = Image.open(img_path)
    name = ''
    face, prob = mtcnn(img, return_prob=True)
    if(face == None):
        return ('Unknown', -1, prob)
    emb = resnet(face.unsqueeze(0)).detach()

    dist_list = []  # list of matched distances, minimum distance is used to identify the person

    for idx, emb_db in enumerate(embedding_list):
        dist = torch.dist(emb, emb_db).item()
        dist_list.append(dist)

    idx_min = dist_list.index(min(dist_list))
    name = name_list[idx_min]
    if(min(dist_list) > 1):
        name = 'Unknown'
    return (name, dist_list[idx_min], prob)


# Load the cascade
faceCascade = cv2.CascadeClassifier(
    'data/haarcascades/haarcascade_frontalface_default.xml')

# To capture video from webcam.
video_capture = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        grey, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)
    index = 0
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        cv2.imwrite(f'{index}.jpg', face)
        index += 1
    index = 0
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        result = face_match(f'{index}.jpg')
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        if(result[0] == 'Unknown'):
            cv2.putText(frame, f'{" ".join(result[0].split("_"))}', (x + 6, y - 6),
                        font, 1, (0, 255, 0), 1)
        else:
            cv2.putText(frame, f'{" ".join(result[0].split("_"))} distance {round(result[1],2)}', (x + 6, y - 6),
                        font, 1, (0, 255, 0), 1)
        index += 1
    cv2.imwrite('face.jpg', frame)
    cv2.imshow('face.jpg', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
