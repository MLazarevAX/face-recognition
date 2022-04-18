import face_recognition
from PIL import Image, ImageDraw
from numpy.core.multiarray import result_type

img_path = "dataset/Putin5.jpg"


def face_rec():
    obama_face_img = face_recognition.load_image_file('fases/Obama.jpg')
    obama_face_location = face_recognition.face_locations(obama_face_img)
    putin_and_obama_img = face_recognition.load_image_file('fases/Obama2.jpg')
    putin_and_obama_face_location = face_recognition.face_locations(putin_and_obama_img)

    # print(obama_face_location, putin_and_obama_face_location)
    # print(f"Found {len(obama_face_location)} face(s) in this image")
    # print(f"Found {len(putin_and_obama_face_location)} face(s) in this image")

    pil_img2 = Image.fromarray(putin_and_obama_img)
    draw2 = ImageDraw.Draw(pil_img2)

    for (top, right, bottom, left) in putin_and_obama_face_location:
        draw2.rectangle(((left, top), (right, bottom)), outline=(255, 255, 0), width=4)

    del draw2
    pil_img2.save("new_faces/obama_putin_with_rect.jpg")


def extracting_faces(path):
    count = 0
    faces = face_recognition.load_image_file(path)
    faces_locations = face_recognition.face_locations(faces)

    for faces_location in faces_locations:
        top, right, bottom, left = faces_location

        face_img = faces[top:bottom, left:right]
        pil_img = Image.fromarray(face_img)
        pil_img.save(f"new_faces/face_img{count}.jpg")
        count += 1
    return print(f"Found {count} face(s) in this photo")


def compare_faces(img_path1, img_path2):
    img1 = face_recognition.load_image_file(img_path1)
    img1_encodings = face_recognition.face_encodings(img1)[0]
    print(img1_encodings)

    img2 = face_recognition.load_image_file(img_path2)
    img2_encodings = face_recognition.face_encodings(img2)[0]
    print(img2_encodings)

    result = face_recognition.compare_faces([img1_encodings], img2_encodings)
    print(result)


def main():
    # face_rec()
    # extracting_faces(img_path)
    compare_faces(r'F:\Maks\Face recognition\fases\Putin.jpg', r'F:\Maks\Face recognition\fases\Putin2.jpg')


if __name__ == "__main__":
    main()
