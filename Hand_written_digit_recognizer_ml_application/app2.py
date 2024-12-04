import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf



mnist= tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test)  =mnist.load_data()
model = tf.keras.models.load_model('digits2.keras')


loss, accuracy= model.evaluate(X_test,y_test)
print(loss)
print(accuracy)

image_number=1

while os.path.isfile(f"digits/digit{image_number}.png" ):
        img=cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img=np.invert(np.array([img]))
        prediction=model.predict(img)
        print(f"The predicted image number probably is {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        image_number+=1            
        
