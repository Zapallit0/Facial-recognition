import cv2
from random import randrange

#Load some pre-trained data on face frontals from opencv(haar cascade algoritm)
trained_face_data= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#Choose the image to detect faces in
#img= cv2.imread('faces/50.jpg')
webcam=cv2.VideoCapture(0)

while True:
    #Leer el frame actual
    successful_frame_read, frame=webcam.read()
    #Convierte a escala de grises
    grayscaled_img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detector de caras
    #nos da los rectangulos alrededor de la cara 
    face_coordinates=trained_face_data.detectMultiScale(grayscaled_img)

    #Dibujar rectangulos alrededor de las caras
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(randrange(128,256),randrange(128,256),randrange(128,256)),10)
    
    cv2.imshow('Clever Programmer Face Detector', frame)
    key=cv2.waitKey(1)

    #Para parar el loop y cerrar el app aplastar Q
    if key==81 or key==113:
        break
#Liberar la camara
webcam.release()


# #print(face_coordinates)
# print("Code completed")