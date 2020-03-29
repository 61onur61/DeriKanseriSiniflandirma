
# DeriKanseriSiniflandirma
7 farklı deri kanseri türüne sahip 10000 resimden oluşan veriseti üzerinde seçilen resmin hangi kanser türüne ait olduğunu gösteren uygulama. 10000 verinin 8184 tanesi eğitim 1816 tanesi test verisi olarak kullanılmıştır. İki tane cnn modeli eğitilmiştir. Cnn modelleri ileilgili bilgiler aşağıdadır. 
 
 
Cnn Modeli1 

input_shape = (75,100,3) #75x100 lük renkli resimler 
num_classes = 7  #7 tane deri kanseri çeşidi var   
#flattenden öncesi feature çıkarımı için  
model = Sequential() #32 nörondan oluşan filtresi 3x3 olan aktivasyon fonksiyonu relu olan Conv2D layer                 model.add(Conv2D(32,kernel_size=(3,3),activation="relu",padding="Same",input_shape=input_shape)) 
model.add(Conv2D(32,kernel_size=(3,3),activation="relu",padding="Same")) 
model.add(MaxPool2D(pool_size=(2,2))) #feature çıkarımı için 
model.add(Dropout(0.5)) #overfitting i önlemek için  
model.add(Conv2D(64,kernel_size=(3,3),activation="relu",padding="Same")) 
model.add(Conv2D(64,kernel_size=(3,3),activation="relu",padding="Same")) 
model.add(MaxPool2D(pool_size=(2,2))) model.add(Dropout(0.5))  
model.add(Flatten()) model.add(Dense(128,activation="relu")) 
model.add(Dropout(0.5)) 
model.add(Dense(num_classes,activation="softmax")) #output layer nöron sayısı= 7 çünkü 7 tane deri kanseri türü var. 
model.summary()                      #aktivasyon softmax çünkü 7 clasın en yüksek oranda çıkanını gösterecek.                                        #summary modelin özetini çıkarır. 
optimizer = Adam(lr=0.0001)            #optimizasyon algoritması Adam learning rate içerir. Modelin öğrenme hızı. model.compile(optimizer = optimizer,loss="categorical_crossentropy",metrics=["accuracy"])                        #loss fonksiyonu categorical_crossentropy değerlendierme metriği accuracy 
epochs = 15 #8184 resimin modeli eğitmek için kaç kez kullanılacağı 
batch_size = 25 #verinin eğitim için kaçar kaçar modele alınacağı.     
#shuffle = True ise eğitim verileri random olarak alınır. 
history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True)  

loss değeri = 0.5513 accuracy değeri = 0.8006   


Cnn Modeli2 

input_shape = (75,100,3) #75x100 lük renkli resimler 
num_classes = 7  #7 tane deri kanseri çeşidi var   #flattenden öncesi feature çıkarımı için  
model = Sequential() #32 nörondan oluşan filtresi 3x3 olan aktivasyon fonksiyonu relu olan Conv2D layer 
model.add(Conv2D(64,kernel_size=(3,3),activation="relu",padding="Same",input_shape=input_shape)) 
model.add(Conv2D(64,kernel_size=(3,3),activation="relu",padding="Same")) 
model.add(Conv2D(64,kernel_size=(3,3),activation="relu",padding="Same")) 
model.add(Conv2D(64,kernel_size=(3,3),activation="relu",padding="Same")) 
model.add(MaxPool2D(pool_size=(2,2))) #feature çıkarımı için 
model.add(Dropout(0.25)) #overfitting i önlemek için  
model.add(Conv2D(64,kernel_size=(3,3),activation="relu",padding="Same")) 
model.add(Conv2D(64,kernel_size=(3,3),activation="relu",padding="Same")) 
model.add(Conv2D(64,kernel_size=(3,3),activation="relu",padding="Same")) 
model.add(Conv2D(64,kernel_size=(3,3),activation="relu",padding="Same")) 
model.add(MaxPool2D(pool_size=(2,2))) model.add(Dropout(0.25))  
model.add(Flatten()) model.add(Dense(128,activation="relu")) 
model.add(Dropout(0.25)) 
model.add(Dense(num_classes,activation="softmax")) #output layer nöron sayısı= 7 çünkü 7 tane deri kanseri türü var. 
model.summary()                      #aktivasyon softmax çünkü 7 clasın en yüksek oranda çıkanını gösterecek.                                        #summary modelin özetini çıkarır. 
optimizer = Adam(lr=0.00001)            #optimizasyon algoritması Adam learning rate içerir. Modelin öğrenme hızı. model.compile(optimizer = optimizer,loss="categorical_crossentropy",metrics=["accuracy"])                        #loss fonksiyonu categorical_crossentropy değerlendierme metriği accuracy 
epochs = 20 #8184 resimin modeli eğitmek için kaç kez kullanılacağı 
batch_size = 25 #verinin eğitim için kaçar kaçar modele alınacağı.     
#shuffle = True ise eğitim verileri random olarak alınır. 
history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True)

loss değeri = 0.1821 accuracy değeri = 0.9316

Datai Team Teşekkürler...

