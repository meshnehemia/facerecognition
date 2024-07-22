import cv2
import numpy as npy
import face_recognition as face_rec
import os

testpath = 'images_test'
tstlist = os.listdir(testpath)
def resize(img, size) :
        width = int(img.shape[1]*size)
        height = int(img.shape[0] * size)
        dimension = (width, height)
        return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)
for clt in tstlist :
    test_path = f'{testpath}/{clt}'  
    joshik = face_rec.load_image_file(test_path)
    joshik = cv2.cvtColor(joshik, cv2.COLOR_BGR2RGB)
    joshik = resize(joshik, 0.50)
    identify = face_rec.face_locations(joshik)
    while identify:
        faceLocation_joshik = identify[0]
        encode_joshik = face_rec.face_encodings(joshik)[0]
        cv2.rectangle(joshik, (faceLocation_joshik[3], faceLocation_joshik[0]), (faceLocation_joshik[1], faceLocation_joshik[2]), (255, 0, 255), 3)
        detect = False
        path = 'images_database'
        name = 'not recognized'
        myList = os.listdir(path)
        for cl in myList :
            if detect:
                break
            else: 
                joshik_test = face_rec.load_image_file(f'{path}/{cl}')   
                joshik_test = resize(joshik_test, 0.50)
                joshik_test = cv2.cvtColor(joshik_test, cv2.COLOR_BGR2RGB)    
                faceLocation_joshiktest = face_rec.face_locations(joshik_test)[0]
                encode_joshiktest = face_rec.face_encodings(joshik_test)[0]
                results = face_rec.compare_faces([encode_joshik], encode_joshiktest)
                if results[0]:
                    name = os.path.splitext(cl)[0]
                    print(f'hello {os.path.splitext(cl)[0]}')
                    detect = True
                else :
                    print(f'you are not {os.path.splitext(cl)[0]}')
        break
    cv2.putText(joshik, f'{name}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 2 )
    cv2.imshow('main_img', joshik)
    name = 'not recognized'
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    