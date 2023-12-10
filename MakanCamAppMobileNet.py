import cv2

import tensorflow as tf
import numpy as np

cap = cv2.VideoCapture(0)

new_model = tf.keras.models.load_model(
    'makan_mobilenet.h5')

new_model.summary()

while True:
    ret, frame = cap.read()

    if not ret:
        break
     
    text="Dunno"
    
    input_size = (224,224)
    resized_frame = cv2.resize(frame, input_size)
    normalized_frame = (resized_frame/255.0-0.5)/0.5

    batch_frame = np.expand_dims(normalized_frame, axis=0)
    predictions = new_model.predict(batch_frame)

    predicted_classes = np.argmax(predictions, axis=-1)
    predicted_class =  predicted_classes[0]

    #cv2.namedWindow("",cv2.WINDOW_NORMAL)
    
    print(f"Predicted class {predicted_class}")


    food=""
    if (max(predictions[0])>0.6):
    
        if predicted_class == 0:
            print("Bak chor mee")
            food ="Bak chor mee"
        
        elif predicted_class == 1:
            print("kaya toast")
            food="Kaya toast"

        elif predicted_class == 2:
            print("Laksa")
            food="Laksa"
        
        elif predicted_class == 3:
            print("chicken rice")
            food="Chicken rice"

        elif predicted_class == 4:
            print("Curry Puff")
            food="Curry Puff"

        elif predicted_class == 5:
            print("fish soup")
            food="Fish Soup"
    else: 
        food ="Cannot Identify"

    if food != "Cannot Identify":
        text = food +" : "+str(max(predictions[0]))
    
    position = (50,50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    thickness = 2

    cv2.putText(frame, text, position, font, font_scale,font_color, thickness)
   # print(predictions)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow("Makan", frame)
    
cap.release()
cv2.destroyAllWindows()
