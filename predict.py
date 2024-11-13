import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow as tf
from PIL import ImageFont, ImageDraw, Image
import google.protobuf


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

model = tf.keras.models.load_model("Our_Model.h5")

#labels=['ا','ب','ت','ث','ج','ح','خ','د','ذ','ر','ز','س','ش','ص','ض','ط','ظ','ع','غ','ف','ق','ك','ل','م','ن','و','ي','ى','ة','ال','لا']
labels=['aleff', 'bb', 'thal', 'seen', 'sheen', 'saad', 'dhad', 'dal', 'dha', 'ain', 'ghain', 'fa', 'taa', 'gaaf', 'kaaf', 'ta', 'meem', 'nun', 'ha', 'waw', 'yaa', 'ya', 'toot', 'thaa', 'al', 'la', 'haa', 'jeem', 'kha', 'zay', 'laam', 'ra']
# font = ImageFont.truetype(r"C:\Windows\Fonts\simpfxo.ttf", 32)

offset = 20
imageSize = 64

letters = []  # initialize letters list

while True:
    success, img = cap.read()
    imgoutput = img.copy()
    hands, ing = detector.findHands(img, draw=True)

    try:
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imageWhite = np.ones((imageSize, imageSize, 3), np.uint8) * 255

            imgcrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

            imageCropShape = imgcrop.shape

            aspectRatio = h/w

            if aspectRatio > 1:
                k = imageSize / w
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgcrop, (wCal, imageSize))
                imageResizeShape = imgResize.shape
                wGap = math.ceil((imageSize-wCal)/2)
                imageWhite[:, wGap:wCal+wGap] = imgResize

                img_array = np.array(imageWhite)
                img_array = np.expand_dims(img_array, axis=0)

                # Predict the class of the new image
                prediction = model.predict(img_array)
                class_index = np.argmax(prediction)

                if cv2.waitKey(1) & 0xFF == ord('a'):  # check if 'a' is pressed
                    letters.append(labels[class_index])  # add predicted letter to list

            else:
                k = imageSize / h
                hCal = math.ceil(k*h)
                imgResize = cv2.resize(imgcrop, (imageSize, hCal))
                imageResizeShape = imgResize.shape
                hGap = math.ceil((imageSize-hCal)/2)
                imageWhite[hGap:hCal+hGap, :] = imgResize

                img_array = np.array(imageWhite)
                img_array = np.expand_dims(img_array, axis=0)

                # Predict the class of the new image
                prediction = model.predict(img_array)
                class_index = np.argmax(prediction)

                if cv2.waitKey(1) & 0xFF == ord('a'):  # check if 'a' is pressed
                    letters.append(labels[class_index])  # add predicted letter to list

            cv2.putText(imgoutput, labels[class_index], (x,y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,255), 2)
            cv2.rectangle(imgoutput, (x-offset,y-offset), (x+w+offset,y+h+offset), (255,0,255), 4)
            # cv2.imshow("ImageCrop", imgcrop)
            # cv2.imshow("ImageWhite", imageWhite)

    except:
        pass

    cv2.putText(imgoutput, ''.join(letters), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)  # display predicted letters on screen
    cv2.imshow("Image", imgoutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to quit program
        break

cv2.destroyAllWindows()
cap.release()