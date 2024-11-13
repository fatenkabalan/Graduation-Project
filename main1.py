from kivy.app import App
from kivy.core.window import Window
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.image import AsyncImage, Image
from kivy.uix.screenmanager import Screen, ScreenManager, CardTransition
from kivy.clock import Clock
from kivy.graphics import Rectangle
from kivy.uix.boxlayout import BoxLayout
import tkinter as tk
import os
import sys
from kivy.graphics.texture import Texture
from kivy.uix.widget import Widget
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow as tf
from kivy.properties import StringProperty, ObjectProperty
import threading
Window.clearcolor=(0,0,63/255)
Window.size=(1100,600)

class ScreenManagment(ScreenManager):
    pass

class Screen1(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)

class Screen2(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)
    def open_camera(self):
        print("switching to screen3")
        self.manager.current = "Screen3"
    def close_and_open_tkinter(self):
        App.get_running_app().stop()
        Window.close() 
        os.system('python3 tek.py')
        sys.exit()
    def open_image(self):
        img = cv2.imread('images/sign_language.png',1)
        cv2.imshow('sign_language',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()       
class Screen3(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.label = Label(text='')
        self.capture = cv2.VideoCapture(0)
        self.image=Image()
        #Clock.schedule_interval(self.load_video, 1.0 / 30.0)
        main_layout = BoxLayout(orientation='vertical')
        layout=BoxLayout(orientation="horizontal")
        second_layout = BoxLayout(orientation='vertical',size_hint = (0.4, 1))

        img=Image(source="images/logo.png",size_hint=(.3,.3),pos_hint={'center_x':0.5})
        main_layout.add_widget(img)
        main_layout.add_widget(layout) 
        main_layout.add_widget(Label(text='', size_hint=(0.2, 0.2))) 
        layout.add_widget(self.image)
        layout.add_widget(second_layout)
        second_layout.add_widget(Label(text="The predicted Letter:",color=(212/255, 180/255, 20/255,1),font_name='Ubuntu-BoldItalic.ttf',font_size=16))
        second_layout.add_widget(self.label)
        second_layout.add_widget(Label(text="The Word:",color=(212/255, 180/255, 20/255,1),font_name='Ubuntu-BoldItalic.ttf',font_size=16))
        second_layout.add_widget(Label(text="",color=(212/255, 180/255, 20/255,1),font_name='Ubuntu-BoldItalic.ttf',font_size=16))  
        second_layout.add_widget(Label(text="The Sentence:",color=(212/255, 180/255, 20/255,1),font_name='Ubuntu-BoldItalic.ttf',font_size=16))
        second_layout.add_widget(Label(text="",color=(212/255, 180/255, 20/255,1),font_name='Ubuntu-BoldItalic.ttf',font_size=16))
        self.image.bind(on_touch_down=self.camera)

        self.add_widget(main_layout)

    # def load_video(self, *args):
    #     ret, frame = self.capture.read()
    #     #frame initialize
    #     self.image_frame=frame
    #     buffer = cv2.flip(frame, 0).tostring()
    #     texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
    #     texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
    #     self.image.texture=texture
    def camera(self, *args):
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
            img = cv2.resize(img, (720, 400))
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
                        self.label.text = labels[class_index]
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
                    self.label.text = labels[class_index]
                    print(labels[class_index])
                    cv2.putText(imgoutput, labels[class_index], (x,y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,255), 2)
                    cv2.rectangle(imgoutput, (x-offset,y-offset), (x+w+offset,y+h+offset), (20, 180, 212), 4)
                    # cv2.imshow("ImageCrop", imgcrop)
                    # cv2.imshow("ImageWhite", imageWhite)
                    

            except:
                pass
            cv2.putText(imgoutput, ''.join(letters), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)  # display predicted letters on screen
            cv2.imshow("SignalTranslator", imgoutput)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to quit program
                break
        
        cv2.destroyAllWindows()
        cap.release()
    
        


        
class Main1App(App):

    def build(self):
        
        sm = ScreenManagment(transition = CardTransition())
        self.title='SignalTranslator'
        sm.add_widget(Screen1(name= "Screen1"))
        sm.add_widget(Screen2(name= "Screen2"))
        sm.add_widget(Screen3(name= "Screen3"))
        Clock.schedule_once(lambda dt: sm.switch_to(Screen2(name = 'Screen2')), 30)
        
        return sm
if __name__ == "__main__":
    Main1App().run()

