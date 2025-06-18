import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 固定全局类别列表（13类）
class_list = [
    'No entry', 'No passing', 'No vechiles',
    'output_20', 'output_30', 'output_40', 'output_50', 'output_70',
    'output_80', 'output_100', 'output_120',
    'Stop', 'Yield'
]

data_path = "/home/nvidia/spd/js/js11"
model_path = "/home/nvidia/git/tsfl/model/model_js11_mask.h5"

# 图像预处理函数
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

# 加载单个数据子集
def load_data(data_path):
    images, labels = [], []
    class_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    for class_name in class_folders:
        if class_name not in class_list:
            continue  # 跳过未知类
        class_index = class_list.index(class_name)
        img_dir = os.path.join(data_path, class_name)
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(preprocess(img))
                labels.append(class_index)
    return np.array(images), np.array(labels)

# 构建模型
def create_model(num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation=None, input_shape=(32, 32, 1)),
        tf.keras.layers.ReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation=None),
        tf.keras.layers.ReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Conv2D(64, (3, 3), activation=None),
        tf.keras.layers.ReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation=None),
        tf.keras.layers.ReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=None),
        tf.keras.layers.ReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# 加载数据
X, Y = load_data(data_path)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
Y = to_categorical(Y, num_classes=len(class_list))

# 获取本轮出现的类
local_classes = np.unique(np.argmax(Y, axis=1)).tolist()
class_mask = np.zeros(len(class_list), dtype=np.float32)
for cls in local_classes:
    class_mask[cls] = 1.0
class_mask = tf.constant(class_mask)

# 自定义 loss：局部 mask loss
def masked_categorical_crossentropy(y_true, y_pred):
    y_true_masked = y_true * class_mask
    y_pred_masked = y_pred * class_mask
    return tf.keras.losses.categorical_crossentropy(y_true_masked, y_pred_masked)

# 构建并编译模型
model = create_model(num_classes=len(class_list))
model.compile(optimizer='adam', loss=masked_categorical_crossentropy, metrics=['accuracy'])

# 拆分数据集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=42)

# 模型训练
model.fit(X_train, Y_train, validation_split=0.2, epochs=5, batch_size=32)

# 测试评估
loss, acc = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {acc:.4f}")

model.save(model_path)
