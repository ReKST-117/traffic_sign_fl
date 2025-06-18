import os
import re
import time
import cv2
import numpy as np
import tensorflow as tf
import subprocess
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ========== 配置区域 ==========
recv_dir = '/home/nvidia/git/tsfl/log/'  # 📥 接收日志和模型的目录
client_name = 'jetson'              # ✅ 当前终端名称
class_list = [
    'No entry', 'No passing', 'No vechiles',
    'output_20', 'output_30', 'output_40', 'output_50', 'output_70',
    'output_80', 'output_100', 'output_120',
    'Stop', 'Yield'
]
server_ip = '192.168.100.143'         # ✅ 云端 IP
server_user = 'seeed'               # ✅ 云端用户名
server_dir = '/home/seeed/fl/receive'  # ✅ 云端 UPLOAD_DIR
local_done = set()                  # 已完成训练轮次记录

# ========== 模型结构 ==========
def create_model(num_classes):
    return tf.keras.models.Sequential([
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

# ========== 预处理函数 ==========
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    return img / 255.0

def load_data(data_path):
    images, labels = [], []
    class_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    for class_name in class_folders:
        if class_name not in class_list:
            continue
        label_idx = class_list.index(class_name)
        folder = os.path.join(data_path, class_name)
        for img_file in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, img_file))
            if img is not None:
                images.append(preprocess(img))
                labels.append(label_idx)
    return np.array(images), np.array(labels)

# ========== 主循环 ==========
print("✅ Monitoring cloud logs，waiting for new training command...")
while True:
    time.sleep(10)
    txt_files = [f for f in os.listdir(recv_dir) if f.endswith('.txt')]

    for fname in txt_files:
        fpath = os.path.join(recv_dir, fname)
        with open(fpath, 'r') as f:
            content = f.read().strip()

        match = re.search(r'js(\d\d)', content)
        if not match:
            continue
        round_id = int(match.group(1)[1])
        if round_id in local_done:
            continue

        print(f"\n📌 Executing Round {round_id} Trainning...")

        data_path = f'/home/nvidia/spd/js/js{round_id}'
        X, Y = load_data(data_path)
        X = X.reshape(-1, X.shape[1], X.shape[2], 1)
        Y = to_categorical(Y, num_classes=len(class_list))

        # ========== 局部掩码 ==========
        local_classes = np.unique(np.argmax(Y, axis=1)).tolist()
        mask = np.zeros(len(class_list), dtype=np.float32)
        for cls in local_classes:
            mask[cls] = 1.0
        class_mask = tf.constant(mask)

        def masked_cce(y_true, y_pred):
            return tf.keras.losses.categorical_crossentropy(y_true * class_mask, y_pred * class_mask)

        # ========== 构建模型 ==========
        model = create_model(len(class_list))
        agg_model_path = os.path.join(recv_dir, f'global_round{round_id - 1}.h5')
        if os.path.exists(agg_model_path):
            model.load_weights(agg_model_path)

        model.compile(optimizer='adam', loss=masked_cce, metrics=['accuracy'])

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)
        model.fit(X_train, Y_train, validation_split=0.2, epochs=5, batch_size=32)
        loss, acc = model.evaluate(X_test, Y_test)

        # ========== 保存并上传 ==========
        model_name = f'client_{client_name}_round{round_id}.h5'
        log_name = f'client_{client_name}_round{round_id}.txt'
        model.save(model_name)

        with open(log_name, 'w') as f:
            f.write(f'Client {client_name} finished round {round_id}, accuracy: {acc:.4f}\n')

        subprocess.run(['scp', model_name, f'{server_user}@{server_ip}:{server_dir}/'])
        subprocess.run(['scp', log_name, f'{server_user}@{server_ip}:{server_dir}/'])

        print(f"✅ Round {round_id} 上传完成，准确率：{acc:.4f}")
        local_done.add(round_id)
