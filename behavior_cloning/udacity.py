import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense

lines = []
with open('sample_training_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines[1:]:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'sample_training_data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurements.append(float(line[3]))

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)

model.save('model.h5')
