import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
#load the trained model to classify sign
from keras.models import load_model
model = load_model('C:/Users/BALAJI BALU/Downloads/mini project/leaf detection/obj_reco_1/tst_model.h5')

#dictionary to label all traffic signs class.
classes = { 1:'Healthy',
            2:'Gray_leaf Spot',      
            3:'Northern Leaf Blight',       
            4:'Common Rust',      
             }
                 
#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Leaf Disease classification')
top.configure(background='#CDCDCD')

label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    imag = Image.open(file_path)
    data_x=[]
    #image = image.resize((30,30))
    #pil_image = PIL.Image.open('Image.jpg').convert('RGB') 
    open_cv_image = np.array(imag) 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    img = image.img_to_array(open_cv_image)
    #img = np.expand_dims(img, axis = 0)
    #img /= 255.0
    #ims=tf.convert_to_tensor(img)
    data_x.append(np.asarray(img, dtype = np.int8))
    raw_data=np.asarray(data_x, dtype = np.float32)
    raw_data/=255.0
    pred = model.predict(raw_data)
    print(pred[0])
    sign = classes[np.argmax(pred[0])+1]
    print(sign)
    label.configure(foreground='#011638', text=sign) 
   

def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Leaf Image",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Upload leaf image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))

upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Welcome to Leaf Disease Classification",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()
