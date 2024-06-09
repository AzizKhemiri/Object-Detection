import tensorflow as tf
import numpy as np
import cv2

#charger le modéle pré-entrainé
model = tf.keras.applications.MobileNetV2(weights='imagenet')

#charger l'image
image = cv2.imread('1.jpeg')

#prétatiter l'image
resized = cv2.resize(image, (224,224))
resized = tf.keras.preprocessing.image.img_to_array(resized)
resized = tf.keras.applications.mobilenet_v2.preprocess_input(resized)

#fiare une prédiction 
predictions = model.predict(np.array([resized]))
decoded_predictions = tf.keras.applications.mobilenet_v2.decode.preprocess_input(resized)

#afficher les résultats
for _,label, score in decoded_predictions[0]:
    print(f"Ceci est peut etre {label} : probabilité {score}")
    
