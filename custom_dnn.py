import keras
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Dropout,Activation,Flatten,MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

def build_model(add_dropout,num_of_class,input_shape):
    
    model = Sequential()
    
    # dropout kullanılan model
    if add_dropout==True:
        print("Dropout kullanılan model import edildi...")
        model.add(Conv2D(32, (3, 3), padding='same',input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))
        
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))     
        
        model.add(Conv2D(128, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (5, 5)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))      
        
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.6))
        model.add(Dense(num_of_class))
        model.add(Activation('softmax'))

        # dropout kullanılmayan model
    elif add_dropout==False:
        print("Dropout kullanılmayan model import edildi...")

        model.add(Conv2D(32, (3, 3), padding='same',input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(128, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (5, 5)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(num_of_class))
        model.add(Activation('softmax'))
        
    return model

def data_fit(data_augmentation,train_path,test_path,width,height,batch_size,epoch,model):
    optimizer = keras.optimizers.Adam(lr=0.001)
    
    if data_augmentation==False:
        print("Augmentation yapılmayacak....")
        
        datagen = ImageDataGenerator(rescale=1/255) 
        X_train = datagen.flow_from_directory(train_path,class_mode="categorical",target_size=(width, height),batch_size=batch_size,shuffle=True)
        X_val = datagen.flow_from_directory(test_path,class_mode="categorical",target_size=(width, height),batch_size=batch_size,shuffle=True)
        
        train_step_size = X_train.n//batch_size
        test_step_size = X_val.n//batch_size
        
        
        model.compile(loss="categorical_crossentropy",optimizer=optimizer,metrics=['acc'])

        history = model.fit_generator(generator=X_train,steps_per_epoch=train_step_size, validation_data=X_val ,validation_steps=test_step_size,
                    epochs=epoch)
        
        
        
    elif data_augmentation==True:
        
        print("Augmentation yapılacak...")


        datagen = ImageDataGenerator(
            featurewise_center=True,  
            samplewise_center=True,  
            zca_whitening=True,  
            zca_epsilon=1e-06,  
            rotation_range=5,  
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            fill_mode='nearest',
            horizontal_flip=True,
            vertical_flip=False, 
            rescale=1/255,
          )

        val_datagen = ImageDataGenerator(rescale=1/255) 
        
        X_train = datagen.flow_from_directory(train_path,class_mode="categorical",target_size=(width, height),batch_size=batch_size)
        X_val = datagen.flow_from_directory(test_path,class_mode="categorical",target_size=(width, height),batch_size=batch_size)
        
        train_step_size = X_train.n//batch_size
        test_step_size = X_val.n//batch_size
        
        model.compile(loss="categorical_crossentropy",optimizer=optimizer,metrics=['acc'])

        history = model.fit_generator(
            X_train,
            steps_per_epoch=train_step_size,
            epochs=epoch,
            validation_data=X_val,
            validation_steps=test_step_size,
          )
    
    return history