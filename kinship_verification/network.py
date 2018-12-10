import numpy as np
import imageio
import scipy.io as scyio
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import SGD
from keras.layers import BatchNormalization
from program import Program

path_to_dataset = '/home/vlado24/code/ml_project_fer/KinFaceW-I'  # samo ovo podesiti na početku
meta_names = ['fd_pairs.mat', 'fs_pairs.mat', 'md_pairs.mat', 'ms_pairs.mat']
dirs = ['father-dau/', 'father-son/', 'mother-dau/', 'mother-son/']
coef1 = 0.8  # [0->0.8] train data
coef2 = 0.9  # [0.8 -> 0.9] validation data and [0.9->1] test data
EPOCH_NUM = 5

def create_model():
    model = Sequential()
    model.add(Convolution2D(16, (5, 5), input_shape=(64, 64, 6)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (5, 5)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(256, (5, 5)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(640))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.0001, momentum=0.9, decay=0.005)
    model.compile(optimizer='sgd', loss="categorical_crossentropy", metrics=['accuracy'])

    print(model.summary())
    return model


# ----Read metadata from .mat files-------
meta_data = []
for name in meta_names:
    mat = scyio.loadmat(path_to_dataset + '/meta_data/' + name)['pairs']
    meta_data.append(mat)


# ----Read dataset for network--------
all_images = []
images_path = []
results = []

for category in range(len(meta_data)):
    directory = '/images/' + dirs[category]

    for elements in meta_data[category]:
        parent_image = path_to_dataset + directory + elements[2][0]
        child_image = path_to_dataset + directory + elements[3][0]
        images_path.append([parent_image, child_image])

        parent_image = imageio.imread(parent_image)
        child_image = imageio.imread(child_image)

        image = np.concatenate((parent_image, child_image), axis=2)
        all_images.append(image)

        results.append([elements[0][0][0], elements[1][0][0]])


results = np.array(results)
all_images = np.array(all_images)
all_images = all_images.astype('float32')
all_images /= 255

# or this way
# all_images -= np.mean(all_images, axis=0)
# all_images /= np.std(all_images, axis=0)


# -----Shuffle data---------------
state = np.random.get_state()
np.random.shuffle(all_images)

np.random.set_state(state)
np.random.shuffle(results)

np.random.set_state(state)
np.random.shuffle(images_path)


# ----Prepare dataset for network--------
index1 = int(len(all_images) * coef1)
index2 = int(len(all_images) * coef2)

X_train = all_images[0: index1]
X_valid = all_images[index1: index2]
X_test = all_images[index2:]

results = keras.utils.to_categorical([elem[1] for elem in results], 2)
Y_train = results[0: index1]
Y_valid = results[index1: index2]
Y_test = results[index2:]

# -----Train network------
model = create_model()
model.fit(X_train, Y_train, batch_size=64, epochs=EPOCH_NUM, validation_data=(X_valid, Y_valid), shuffle=True)
model.save('modelSaved_eph5.dat')

# model = load_model('modelSaved_eph5.dat')


# -----Test data-----
n_test_data = 20
correct_cnt = 0
comp_results = []
new_path = images_path[index2: index2 + n_test_data]
for i in range(n_test_data):
    prediction = model.predict(np.array([X_test[i], ]))
    prediction = np.around(prediction)
    real_result = Y_test[i]
    comp_results.append(prediction[0][1])
    print(new_path[i])

    if prediction[0][0] == real_result[0] and prediction[0][1] == real_result[1]:
        correct_cnt += 1


# -------GUI--------
print('Accuracy: ' + str(correct_cnt/n_test_data))
Program(images_path[index2: index2 + n_test_data], [x[1] for x in Y_test[:n_test_data]], comp_results, correct_cnt/n_test_data)





