import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow as tf
from PIL import ImageFont, ImageDraw, Image,ImageTk
import tkinter as tk
#######################################################################
model = tf.keras.models.load_model("Our_Model.h5")
#labels=['ا', 'ب', 'ذ', 'س', 'ش', 'ص', 'ض', 'د', 'ظ', 'ع', 'غ', 'ف', 'ت', 'ق', 'ك', 'ط', 'م', 'ن', 'ه', 'و', 'ي', 'ى', 'ة', 'ث', 'ال', 'لا', 'ح', 'ج', 'خ', 'ز', 'ل', 'ر']
labels=['ا', 'ب', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ت', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي', 'ى', 'ة', 'ث', 'ال', 'لا', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر']



offset = 20
imageSize = 64
def cap():
    global sentence, img, letter ,word 
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    # Initialize variables
    sentence = ""
    word = ""
    
    
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
                    # Make prediction using the model
                    # Predict the class of the new image
                    prediction = model.predict(img_array)
                    class_index = np.argmax(prediction)
                    print(class_index)
                    if key==ord('a'):
                        prediction = model.predict(img_array)
                        class_index = np.argmax(prediction)
                        letter=labels[class_index]
                        word=word+letter
                        # Update labels with values of variables 
                        var2.set(word) 
                        frame2.update_idletasks()
                    
                else:
                    k = imageSize / h
                    hCal = math.ceil(k*h)
                    imgResize = cv2.resize(imgcrop, (imageSize, hCal))
                    imageResizeShape = imgResize.shape
                    hGap = math.ceil((imageSize-hCal)/2)
                    imageWhite[hGap:hCal+hGap, :] = imgResize

                    img_array = np.array(imageWhite)
                    img_array = np.expand_dims(img_array, axis=0)
                    # Make prediction using the model
                    prediction = model.predict(img_array)
                    class_index = np.argmax(prediction)
                    print(class_index)
                    if key==ord('a'):
                        prediction = model.predict(img_array)
                        class_index = np.argmax(prediction)
                        letter=labels[class_index]
                        word=word+letter
                        # Update labels with values of variables 
                        var2.set(word) 
                        frame2.update_idletasks()
                    # Predict the class of the new image
                cv2.rectangle(imgoutput, (x-offset,y-offset), (x+w+offset,y+h+offset), (20, 180, 212), 4)
                # cv2.imshow("ImageCrop", imgcrop)
                # cv2.imshow("ImageWhite", imageWhite)
            # var.set(" ") 
            # frame2.update_idletasks() 
        except:
            pass
        
        
        # Show camera window
        imgoutput=cv2.resize(imgoutput, (720, 400))
        cv2.imshow('SignalTranslator', imgoutput)
        #cv2.imshow('frame',crop) 
        frame2.pack()     
        
        # Get key from user and perform actions
        key = cv2.waitKey(1)
        
        # Quit
        if key & 0xFF == ord('q'):
            var2.set(word) 
            var3.set(sentence) 
            frame2.update_idletasks()
            break
        
       
        
        # Clear word
        elif key ==ord('c'):
            word="                         "
            var2.set(word)
            frame2.update_idletasks()
            word=""
            # Update labels with values of variables 
            var2.set(word) 
            frame2.update_idletasks()
        elif key ==ord('x'):
            sentence=""
            var3.set(sentence)
            frame2.update_idletasks()
        # Backspace
        elif key ==ord('l'):
            word= word[:-1]
            # Update labels with values of variables 
            var2.set(word)  
            frame2.update_idletasks()
        
        # Add word to sentence
        elif key ==ord(' '):
            sentence = sentence + word + ' '
            word="                          "
            var2.set(word)
            frame2.update_idletasks()
            word=""
            # Update labels with values of variables 
            var2.set(word) 
            var3.set(sentence) 
            frame2.update_idletasks()
            
    # Close camera window
    cap.release()
    cv2.destroyAllWindows()

sentence = ""      # outputs the whole sentence
letter = ''        # outputs a single char
word = ""          # outputs a single word

window = tk.Tk()
window.title('SignalTranslator')
window.geometry("400x400")

bg_image = ImageTk.PhotoImage(Image.open('images/background.jpeg'))
background_label = tk.Label(window, image=bg_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)



button = tk.Button(window, text="Start Capturing" ,command=cap ,fg='gray', bg = "white" , width = 15 ,font=('Ubuntu-BoldItalic.ttf', 16) )    # cap command calls the webcam to start capturing
button.pack(padx=30, pady=30)

frame2=tk.Frame(window, width=250, height=250)     # create a frame with W 250 x H 250




# l1 = tk.Label(frame2, text="The Predicted Letter",font=('Ubuntu-BoldItalic.ttf', 16),fg='gray')
# var = tk.StringVar()    # create a string variable
# var.set(letter)      # set it to "letter"
# l2 = tk.Label(frame2, textvariable = var,font=('Ubuntu-BoldItalic.ttf', 16))   # display var "letter" as l2 
l3 = tk.Label(frame2, text="The word",font=('Ubuntu-BoldItalic.ttf', 16),fg='gray')

var2 = tk.StringVar()
var2.set(word)        # do the same with "word" and store it as a var
l4 = tk.Label(frame2, textvariable = var2,font=('Ubuntu-BoldItalic.ttf', 16))   # display it as l4

l5 = tk.Label(frame2, text="The sentence",font=('Ubuntu-BoldItalic.ttf', 16),fg='gray')

var3 = tk.StringVar()
var3.set(sentence)      # do the same with "sentence" and store it as a var
l6 = tk.Label(frame2, textvariable = var3,font=('Ubuntu-BoldItalic.ttf', 16))   # display it as l6

# l1.pack(padx=10, pady=10)
# l2.pack(padx=10, pady=10)
l3.pack(padx=10, pady=10)
l4.pack(padx=10, pady=10)
l5.pack(padx=10, pady=10)
l6.pack(padx=10, pady=10)

window.mainloop()