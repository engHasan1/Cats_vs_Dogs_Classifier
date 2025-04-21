import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model('cats_vs_dogs_model_vgg16_finetuned.keras')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()


class_names = ['Cat', 'Dog']

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame")
        break

    img = cv2.resize(frame, (150, 150)) 
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    score = prediction[0][0]
    label = class_names[int(score > 0.5)]
    confidence = score if score > 0.5 else 1 - score

  
    text = f"{label}: {confidence:.2f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

  
    cv2.imshow('Cats vs Dogs Classifier', frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()