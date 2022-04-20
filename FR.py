import face_recognition
from PIL import Image, ImageDraw
from numpy.core.multiarray import result_type

img_path = "dataset/Putin5.jpg"


def face_recognition(path_from, path_to):
    putin_and_obama_img = face_recognition.load_image_file(path_from)
    putin_and_obama_face_location = face_recognition.face_locations(putin_and_obama_img)

    pil_img = Image.fromarray(putin_and_obama_img)
    draw = ImageDraw.Draw(pil_img)

    for (top, right, bottom, left) in putin_and_obama_face_location:
        draw.rectangle(((left, top), (right, bottom)), outline=(255, 255, 0), width=4)

    del draw
    pil_img.save(path_to)


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
    # face_rec(img_path)
    # extracting_faces(img_path)
    compare_faces(r'F:\Maks\Face recognition\fases\Putin.jpg', r'F:\Maks\Face recognition\fases\Putin2.jpg')


if __name__ == "__main__":
    main()
