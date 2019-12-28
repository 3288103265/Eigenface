import os

import cv2

from PIL import Image
import numpy as np

from eigenfaces import get_save_eigenfaces


def center_crop(patch):
    x = patch[0] + (patch[2] - 50) // 2
    y = patch[1] + (patch[3] - 50) // 2
    new_patch = (x, y, 50, 50)

    return new_patch

def get_label(patch, project_mat, eigens):
    sample_in_pca = np.dot(project_mat, patch)
    diff = (eigens - sample_in_pca) / 255
    err = np.sum(np.square(diff), axis=0)

    return err.argmin()

def face_recognition(img_dir):
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    src = Image.open(img_dir).convert('L')
    img = np.array(src)
    faces = detector.detectMultiScale(img, 1.1, 15)

    project_mat, mean, imgs = get_save_eigenfaces()
    eigens_in_pca = np.dot(project_mat, imgs)
    # draw the bounding box.
    for face in faces:
        (x, y, w, h) = center_crop(face)
        patch = img[x:x+w, y:y+h].copy()
        patch2 = np.zeros_like(patch)
        cv2.equalizeHist(patch, patch2)
        patch2.resize((2500, 1))
        patch2 = patch2 - mean

        label = get_label(patch2, project_mat, eigens_in_pca)

        cv2.putText(img, 'P'+ str(label), (x+w, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        basename = os.path.basename(img_dir)
        name = basename.split('.')[0]+'.jpg'
        cv2.imwrite('recognition res/'+name, img)

    cv2.imshow('result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --------------------------------------------------
face_recognition('faces/group/smiling/00016.tga')