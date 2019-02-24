import numpy as np
import imageio
import scipy.io as scyio
import keras
from os import listdir
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


path_to_dataset = '/home/vlado24/code/ml_project_fer/KinFaceW-I'  # PODESITI ${PATH} na početku!
meta_names = ['fd_pairs.mat', 'fs_pairs.mat', 'md_pairs.mat', 'ms_pairs.mat']
dirs = ['father-dau/', 'father-son/', 'mother-dau/', 'mother-son/']
coef1 = 0.8  # [0->0.8] train data
coef2 = 0.9  # [0.8 -> 0.9] validation data and [0.9->1] test data
EPOCH_NUM = 50


def create_model():
    """
    This method create neural network model.
    :return: model
    """

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


def train_network():
    """
    This method prepare dataset for network and train/load neural network.
    :return:
    """

    global index2, X_test, Y_test, model, images_path

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
    # model = create_model()
    # model.fit(X_train, Y_train, batch_size=64, epochs=EPOCH_NUM, validation_data=(X_valid, Y_valid), shuffle=True)
    # model.save('modelSaved_eph' + str(EPOCH_NUM) + '.dat')

    # -------Load model-------
    model = load_model('modelSaved_eph' + str(EPOCH_NUM) + '.dat')


def test_n_new_pairs(n_test_data, path):
    files = listdir(path)
    files.sort()
    sorted = files[:int(len(files)/2)]
    to_shuffle = files[int(len(files)/2):]
    np.random.shuffle(to_shuffle)

    files = sorted + to_shuffle

    list_of_b = []
    list_of_a = []
    for name in files:
        if '_a' in name:
            list_of_a.append(name)
            continue

        if '_b' in name:
            list_of_b.append(name)
            continue
        #error

    all_images = []
    results = []
    images_path = []

    for a, b in zip(list_of_a, list_of_b):
        parent_image_pth = path + '/' + a
        child_image_pth = path + '/' + b
        images_path.append([parent_image_pth, child_image_pth])

        parent_image = imageio.imread(parent_image_pth)
        child_image = imageio.imread(child_image_pth)
        image = np.concatenate((parent_image, child_image), axis=2)
        all_images.append(image)

        if a.split('_')[0] == b.split('_')[0]:
            results.append([0., 1.])
        else:
            results.append([1., 0.])

    results = np.array(results)
    all_images = np.array(all_images)
    all_images = all_images.astype('float32')
    all_images /= 255

    # -----Shuffle data---------------
    state = np.random.get_state()
    np.random.shuffle(all_images)

    np.random.set_state(state)
    np.random.shuffle(results)

    np.random.set_state(state)
    np.random.shuffle(images_path)

    # ------Test data------------
    num_of_pairs = all_images.shape[0]
    n_test_data = num_of_pairs if n_test_data > num_of_pairs else n_test_data
    correct_cnt = 0
    comp_results = []
    for i in range(n_test_data):
        prediction = model.predict(np.array([all_images[i], ]))
        prediction = np.around(prediction)
        real_result = results[i]
        comp_results.append(prediction[0][0])

        if prediction[0][1] == real_result[0] and prediction[0][0] == real_result[1]:
            correct_cnt += 1

    print('Accuracy: ' + str(correct_cnt / n_test_data))
    return images_path[:n_test_data], [x[1] for x in results[:n_test_data]], comp_results, (correct_cnt / n_test_data)


def test_n_pairs(n_test_data):
    # -----Test data-----
    correct_cnt = 0
    comp_results = []
    new_path = images_path[index2: index2 + n_test_data]
    for i in range(n_test_data):
        prediction = model.predict(np.array([X_test[i], ]))
        prediction = np.around(prediction)
        real_result = Y_test[i]
        comp_results.append(prediction[0][1])
        print(new_path[i][0][-12:] + ' <=> ' + new_path[i][1][-12:])

        if prediction[0][0] == real_result[0] and prediction[0][1] == real_result[1]:
            correct_cnt += 1

    # -------GUI--------
    print('Accuracy: ' + str(correct_cnt / n_test_data))
    return images_path[index2: index2 + n_test_data], [x[1] for x in Y_test[:n_test_data]], comp_results, (correct_cnt / n_test_data)