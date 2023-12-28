from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from tensorflow.keras.utils import get_file
import os
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation

imagegen = ImageDataGenerator(rescale=1./255., rotation_range=30, horizontal_flip=True, validation_split=0.1)

path_to_zip = get_file('weatherData.zip', origin="https://static.junilearning.com/ai_level_2/weatherData.zip", extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'weatherData')
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')

train_generator = imagegen.flow_from_directory(train_dir, class_mode="categorical", shuffle=True, batch_size=128, target_size=(224, 224))
validation_generator = imagegen.flow_from_directory(validation_dir, class_mode="categorical", shuffle=True, batch_size=128, target_size=(224, 224))
test_generator = imagegen.flow_from_directory(test_dir, class_mode="categorical", shuffle=False, batch_size=128, target_size=(224, 224))

model = Sequential()

# 32; This is how many filters (neurons) we have. We have 8 neurons for each of the 4 classes.
# (3,3); This is the size of our filter. The filter has weights which are multiplied by the input values and takes the average.
# input_shape=(224,224,3); This says that our input shape has 3 values; its height and width (224x224px) and its 3 color value (RGB)
#Max Pooling: takes the output of the previous layer and takes the max of the pool -> reduces size of image to emphasize biggest value

#Input Layer
model.add(Conv2D(32,(3,3), input_shape=(224,224,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#Dropout = takes away 25% of the data to teach the AI different pieces to help it understand a general image. Not doing this means it sees one picture over and over which means when it sees another image it gets confused.
model.add(Dropout(0.25))


#Hidden Layer
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))

#Output Layer
model.add(Dense(4, activation='softmax'))

#Tip: Classification = crossentropy (2 outputs = binary, categories = categorical)
#Tip: Optimizer - 'adam' is best for classification, mse for regression
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_generator, epochs=20, validation_data=validation_generator)

accuracy = model.evaluate(test_generator)[1]
print(f'Accuracy: {accuracy}')

example_data_gen = ImageDataGenerator(rescale=1/255)
example_generator = example_data_gen.flow_from_directory(PATH, target_size=(224,224), classes=['examples'], shuffle=False)
example_generator.reset()
predictions = model.predict(example_generator)
print(predictions)

predicted_classes = []
for prediction in predictions:
  predicted_class = -1
  maximum = -1
  for i, output in enumerate(prediction):
    if output > maximum:
      maximum = output
      predicted_class = i
  predicted_classes.append(predicted_class)
for i, filename in enumerate(example_generator.filenames):
  print(f'Filename: {filename}')
  print(list(train_generator.class_indices.items())[i][0])




