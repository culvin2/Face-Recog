import face_recognition
import os
import cv2


KNOWN_FACES_DIR = "known_faces"#ambil data dari folder datamuka
TOLERANCE = 0.5  # toleransi detect muka mirip si a, tp bner gk nih?
FRAME_THICKNESS = 3 #tebel garis kotak
FONT_THICKNESS = 1 # tebel font
MODEL = "hog" # algoritma baca mukanya

video = cv2.VideoCapture(0)#ngambil input dari webcam 0 = webcam bawaan , kl mw video / image tnggal masukkin nama

print("Loading Face .....")

known_faces = []
known_names =[]

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)

while True:

    ret, image = video.read()#ngubah video jadi image

    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)

    for face_encoding, face_location in zip(encodings, locations):#ngebandingin sama data yang udh ada
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f"Match found: {match}")
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            color = [0, 255, 0]
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)#buat kotak di daerah muka

            #buat label nama
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2]+22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, str(match), (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, (200,200,200),FONT_THICKNESS)

    cv2.imshow(filename, image)#nampilin muka
    if cv2.waitKey(1) & 0xFF ==ord("q"):#tombol exit dengan q
        break

#release holder di webcam laptop/computer
video.release()
cv2.destroyAllWindows()