import cv2
import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import load_model

def preprocess():
    fruits = ['apple', 'banana', 'cherry', 'grapes', 'kiwi', 'mango', 'orange', 'strawberry']

    for i in range (0, len(fruits)):
        dir = f'input-images/{fruits[i]}-fruit'
        files = os.listdir(dir)
        j = 1

        for file in files:
            image_name = dir + '/' + file
            image = cv2.imread(image_name)

            # Preproccess Image
            gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray_scale, (13, 13), 0)
            resized = cv2.resize(blurred, (256, 256))

            cv2.imwrite(f'processed-images/{fruits[i]}-fruit/image_{j}.jpg', resized)
            j += 1
            # cv2.imshow(fruits[i], resized)
            # cv2.waitKey(100)
            # cv2.destroyAllWindows()

def train():
    directory = 'processed-images'
    data_set = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='int',
        batch_size=32,
        image_size=(256, 256),
        shuffle=True
    ) 

    model = models.Sequential()
    
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 3)))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(32, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(8, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(data_set, epochs=10)

    model.save('fruit_classifier_model.keras')

def test():
    model = tf.keras.models.load_model('fruit_classifier_model.keras')

    image = cv2.imread('test-images/test-banana.jpg')
    resized = cv2.resize(image, (256, 256))  
    resized = resized.reshape(1, 256, 256, 3)  # add batch dimension

    predictions = model.predict(resized)
    predicted_class = tf.argmax(predictions[0]).numpy()

    fruit_classes = ['apple', 'banana', 'cherry', 'grapes', 'kiwi', 'mango', 'orange', 'strawberry']
    print(f"Predicted fruit: {fruit_classes[predicted_class]}")

def main():
    # preprocess()
    # train()
    test()

main()