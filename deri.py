#libraries
import tkinter as tk 
from tkinter import ttk 
from tkinter import filedialog
from tkinter import messagebox

from PIL import ImageTk,Image

import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import seaborn as sns 
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical 
from keras.models import Sequential 
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D
from keras.models import load_model
from keras.optimizers import Adam

#data read 
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")



#data information
train.info()
test.info()

#visualization
#sns.countplot(x = "dx" , data = skin_df)



#%% preprocessing
#data_folder_name = "train/"
#
#ext = ".jpg"
#
#train["path"] = [data_folder_name + i + ext for i in train["image_id"]]
#
#train["image"] = train["path"].map(lambda x: np.asarray(Image.open(x).resize((100,75))))
#
#plt.imshow(train["image"][0])
#
#train["dx_idx"] = pd.Categorical(train["dx"]).codes
#
#train.to_pickle("train.pkl")

#%% preprocessing
#data_folder_name = "test/"
#
#ext = ".jpg"
#
#test["path"] = [data_folder_name + i + ext for i in test["image_id"]]
#
#test["image"] = test["path"].map(lambda x: np.asarray(Image.open(x).resize((100,75))))
#
#plt.imshow(test["image"][0])
#
#test["dx_idx"] = pd.Categorical(test["dx"]).codes
#
#test.to_pickle("test.pkl")
#%% load pkl 
train = pd.read_pickle("train.pkl")
test = pd.read_pickle("test.pkl")
#%% standardization 
x_train = np.asarray(train["image"].tolist())
x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)
x_train = (x_train - x_train_mean)/x_train_std


#one hot encoding 
y_train = to_categorical(train["dx_idx"],num_classes= 7)

test_y = np.asarray(test["image"].tolist())
y_train_mean = np.mean(test_y)
y_train_std = np.std(test_y)
test_y = (test_y - y_train_mean)/y_train_std

#%%CNN
# input_shape = (75,100,3) #75x100 lük renkli resimler
# num_classes = 7  #7 tane deri kanseri çeşidi var
#   #flattenden öncesi feature çıkarımı için 
# model = Sequential() #32 nörondan oluşan filtresi 3x3 olan aktivasyon fonksiyonu relu olan Conv2D layer
# model.add(Conv2D(32,kernel_size=(3,3),activation="relu",padding="Same",input_shape=input_shape))
# model.add(Conv2D(32,kernel_size=(3,3),activation="relu",padding="Same"))
# model.add(Conv2D(32,kernel_size=(3,3),activation="relu",padding="Same"))
# model.add(Conv2D(32,kernel_size=(3,3),activation="relu",padding="Same"))
# model.add(MaxPool2D(pool_size=(2,2))) #feature çıkarımı için
# model.add(Dropout(0.25)) #overfitting i önlemek için

# model.add(Conv2D(64,kernel_size=(3,3),activation="relu",padding="Same"))
# model.add(Conv2D(64,kernel_size=(3,3),activation="relu",padding="Same"))
# model.add(Conv2D(64,kernel_size=(3,3),activation="relu",padding="Same"))
# model.add(Conv2D(64,kernel_size=(3,3),activation="relu",padding="Same"))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(128,activation="relu"))
# model.add(Dropout(0.25))
# model.add(Dense(num_classes,activation="softmax")) #output layer nöron sayısı= 7 çünkü 7 tane deri kanseri türü var.
# model.summary()                      #aktivasyon softmax çünkü 7 clasın en yüksek oranda çıkanını gösterecek.
#                                       #summary modelin özetini çıkarır.
# optimizer = Adam(lr=0.0001)            #optimizasyon algoritması Adam learning rate içerir. Modelin öğrenme hızı.
# model.compile(optimizer = optimizer,loss="categorical_crossentropy",metrics=["accuracy"])
#                       #loss fonksiyonu categorical_crossentropy değerlendierme metriği accuracy
# epochs = 20 #8184 resimin modeli eğitmek için kaç kez kullanılacağı
# batch_size = 25 #verinin eğitim için kaçar kaçar modele alınacağı.

#   #shuffle = True ise eğitim verileri random olarak alınır.
# history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True)

# model.save("CNN1.h5")


#%%model load 
model1 = load_model("CNN1.h5")
model2 = load_model("CNN2.h5")
#%%prediction 
#index = 7
#y_pred = model1.predict(test_y[index].reshape(1,75,100,3))
#y_pred_class = np.argmax(y_pred , axis=1)
#print(y_pred)
#print(y_pred_class)

#y_pred2 = model2.predict(x_train[index].reshape(1,75,100,3))
#y_pred_class2 = np.argmax(y_pred2 , axis=1)


#%%
window = tk.Tk()
window.geometry("1100x650")
window.wm_title("Deri Kanseri Sınıflandırma")

##global variables 
img_name = ""
count=0
img_jpg = ""

#frames 
frame_left = tk.Frame(window,width=540,height=640,bd=2)
frame_left.grid(row=0,column=0)

frame_right = tk.Frame(window,width=540,height=640,bd=2)
frame_right.grid(row=0,column=1)

frame1 = tk.LabelFrame(frame_left,text="Resim",width=540,height=500)
frame1.grid(row=0,column=0)

frame2 = tk.LabelFrame(frame_left,text="Model ve Kaydet",width=540,height=140)
frame2.grid(row=1,column=0)

frame3 = tk.LabelFrame(frame_right,text="Özellikler",width=270,height=640)
frame3.grid(row=0,column=0)

frame4 = tk.LabelFrame(frame_right,text="Sonuç",width=270,height=640)
frame4.grid(row=0,column=1,padx=10)

#frame1
def imageResize(img):
    
    basewidth = 500
    wpercent = (basewidth/float(img.size[0])) #1000 x 1200
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize),Image.ANTIALIAS)
    
    return img
def openImage():
    
    global img_name
    global count
    global img_jpg
    
    count+=1
    if count != 1:
        messagebox.showinfo(title="Warning",message="Lütfen bir tane resim seçiniz")
    else:
        img_name = filedialog.askopenfilename(initialdir="C:\Desktop" , title= "Resim dosyasını seçiniz")
        
        img_jpg = img_name.split("/")[-1].split(".")[0]
        #image label
        tk.Label(frame1,text=img_jpg,bd=3).pack(pady=10)
        
        #open and show image
        img = Image.open(img_name)
        img = imageResize(img)
        img = ImageTk.PhotoImage(img)
        panel = tk.Label(frame1,image=img)
        panel.image = img 
        panel.pack(padx = 15 , pady = 10)
        
        #image features 
        data = pd.read_csv("test.csv")
        cancer = data[data.image_id == img_jpg]
        
        for i in range(cancer.size):
            x = 0.5
            y = (i/10)/2
            tk.Label(frame3,font=("Times",12),text=cancer.iloc[0,i]).place(relx = x , rely = y)
            
menubar = tk.Menu(window)
window.config(menu=menubar)
file = tk.Menu(menubar)
menubar.add_cascade(label="Dosya",menu=file)
file.add_command(label="Aç",command=openImage)

#frame3 
def classification():
    
    if img_name != "" and models.get() != "":
        
        #model selection 
        if models.get() == "CNN1":
            classification_model = model1
        elif models.get() == "CNN2":
            classification_model = model2
        
        z = test[test.image_id == img_jpg]
        z = z.image.values[0].reshape(1,75,100,3)
        
        z = (z - x_train_mean)/x_train_std
        h = classification_model.predict(z)[0]
        h_index = np.argmax(h)
        predicted_cancer = list(train.dx.unique())[h_index]
        
        for i in range(len(h)):
            x = 0.5
            y = (i/10)/2
            
            if i != h_index:
                tk.Label(frame4,text = str(h[i])).place(relx=x , rely = y)
            else:
                tk.Label(frame4,bg = "green",text = str(h[i])).place(relx=x , rely = y)
                
        if chvar.get() == 1:
             
             val = entry.get()
             entry.config(state = "disabled")
             path_name = val + ".txt" 
             
             save_txt = img_name + "--" + str(predicted_cancer)
             
             text_file = open(path_name,"w")
             text_file.write(save_txt)
             text_file.close()
        else:
              print("Kaydetme seçeneği seçilmedi")
    else:
        messagebox.showinfo(title="Warning",message="Resim ve Model seçiniz")

columns = ["lesion_id","image_id","dx","dx_type","age","sex","localization"]

for i in range(len(columns)):
    x = 0.05
    y = (i/10)/2
    tk.Label(frame3,font=("Times",12),text=columns[i]+": ").place(relx = x , rely = y)

classify_button = tk.Button(frame3,bg="red",bd=4,font = ("Times",13),activebackground="blue",text="Sınıflandır", command = classification)
classify_button.place(relx=0.1,rely=0.5)


#frame4 
labels = ["akiec","bcc","bkl","df","mel","nv","vasc"]

for i in range(len(labels)):
    x = 0.05
    y = (i/10)/2
    tk.Label(frame4,font=("Times",12),text=labels[i]+"::: ").place(relx = x , rely = y)


#frame2 
#combobox
model_selection_label = tk.Label(frame2,text="Model Seçiniz")
model_selection_label.grid(row=0,column=0,padx=5)

models = tk.StringVar()
model_selection = ttk.Combobox(frame2,textvariable=models,values=("CNN1","CNN2"),state = "readonly")
model_selection.grid(row=0,column=1,padx=5)

#check box
chvar = tk.IntVar()
chvar.set(0)
xbox = tk.Checkbutton(frame2,text="Sınıflandırma sonucunu kaydet",variable = chvar)
xbox.grid(row=1,column=0,pady=5)

#☺entry
entry = tk.Entry(frame2,width = 23)
entry.insert(string= "Dosya Adı",index=0)
entry.grid(row=1,column=1,pady=5)




















window.mainloop()













