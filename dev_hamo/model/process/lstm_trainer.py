import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Dense, LSTM

# 하이퍼 파라미터
LEARNING_RATE = 0.001
N_EPOCHS = 100
N_BATCHS = 100
NUM_CLASSES = 4  # 클래스 개수
DATA_FRAME = 80
CLASS_MODE = 'categorical'

# Use Google Colab
BASE_PATH = '/content/drive/MyDrive'


def data_preprocessing(num):
    # 전체 csv는 80(frame number), 2의 size로 구성 가정
    # 라벨링은 각 동작 csv의 라벨(동작 번호)를 통해

    # 학습데이터 및 모델 경로 설정
    csv_dir = f'/content/drive/MyDrive/gesture_csv_{num}'

    # (80, 2), 프레임*(x,y)로 원하는 학습데이터 형태로 분할
    IMAGE_SIZE = (640, 640)
    INPUT_SHAPE = (DATA_FRAME, 2)  # xy쌍
    data_X, data_y = [], []
    label_list = [dir for dir in os.listdir(csv_dir) if os.path.isdir(os.path.join(csv_dir, dir))]  # [다음, 이전, ...]
    label_list.sort()

    for label, label_name in enumerate(label_list):
        now_label_path = os.path.join(csv_dir, label_name)
        csv_list = [fname for fname in os.listdir(now_label_path) if os.path.splitext(fname)[-1] == '.csv']

        # csv파일 읽어오기
        for csv_name in csv_list:
            df = pd.read_csv(os.path.join(now_label_path, csv_name), header=None)

            # MinMaxScaler가 아닌 캔버스 크기 대비 위치 비율로 data 스케일링 진행
            for i in range(2):
                df[df.columns[i]] = df[df.columns[i]] / IMAGE_SIZE[i % 2]

            data_X.append(df)
            data_y.append(label)

    # label은 OneHotEncoding
    data_y = keras.utils.to_categorical(data_y)

    to_shuff_all = [[x, y] for x, y in zip(data_X, data_y)]
    random.shuffle(to_shuff_all)
    data_X = [tmp[0] for tmp in to_shuff_all]
    data_y = [tmp[1] for tmp in to_shuff_all]

    # train, val, test 분리 7, 2, 1
    train_idx = int(len(data_y) * 0.7)
    val_idx = int(len(data_y) * 0.9)

    X_train, y_train = np.array(data_X[:train_idx]), np.array(data_y[:train_idx])
    X_val, y_val = np.array(data_X[train_idx:val_idx]), np.array(data_y[train_idx:val_idx])
    X_test, y_test = np.array(data_X[val_idx:]), np.array(data_y[val_idx:])

    return X_train, y_train, X_val, y_val, X_test, y_test


def lstm_structure():
    model = tf.keras.Sequential()
    model.add(layers.Input((DATA_FRAME, 2)))
    model.add(layers.LSTM(220, activation='relu', return_sequences=True))
    model.add(layers.LSTM(220, activation='relu', return_sequences=False))
    model.add(layers.Dense(10))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

    return model


def check_model(lstm_model):
    lstm_model = lstm_structure()
    lstm_model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
    lstm_model.summary()



X_train, y_train, X_val, y_val, X_test, y_test = data_preprocessing(num=1)
lstm_model = lstm_structure()
check_model(lstm_model)

rl_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, verbose=1)

# 하이퍼 파라미터
LEARNING_RATE = 0.001
N_EPOCHS = 100
N_BATCHS = 100

N_TRAIN, N_TEST = X_train.shape[0], X_test.shape[0]
steps_per_epoch = N_TRAIN // N_BATCHS
validation_steps = int(np.ceil(N_TEST / N_BATCHS))

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(N_TRAIN).batch(N_BATCHS,
                                                                                              drop_remainder=True).repeat()
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(N_BATCHS)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(N_BATCHS)

history = lstm_model.fit(train_dataset,
                         epochs=N_EPOCHS,
                         steps_per_epoch=steps_per_epoch,
                         validation_data=test_dataset,
                         validation_steps=validation_steps,
                         callbacks=[rl_callback])


""" log
Epoch 00070: ReduceLROnPlateau reducing learning rate to 2.5600002118153498e-09.
Epoch 71/100
8/8 [==============================] - 1s 126ms/step - loss: 0.0646 - accuracy: 0.9962 - val_loss: 0.1005 - val_accuracy: 0.9832
Epoch 72/100
8/8 [==============================] - 1s 126ms/step - loss: 0.0601 - accuracy: 0.9987 - val_loss: 0.1005 - val_accuracy: 0.9832
Epoch 73/100
8/8 [==============================] - 1s 127ms/step - loss: 0.0635 - accuracy: 0.9975 - val_loss: 0.1004 - val_accuracy: 0.9832
Epoch 74/100
8/8 [==============================] - 1s 130ms/step - loss: 0.0634 - accuracy: 0.9975 - val_loss: 0.1004 - val_accuracy: 0.9832
"""

lstm_model.evaluate(test_dataset)

MODEL_PATH = '/content/drive/MyDrive/LSTM_gesture_model'
# # h5형식으로 저장
# model_h5_path = os.path.join(MODEL_PATH, 'lstm_model.h5')
