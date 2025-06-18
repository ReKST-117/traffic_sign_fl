import os
import time
import re
import tensorflow as tf
import numpy as np
import cv2
import subprocess
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ========== Configuration ==========
recv_dir = '/home/nvidia/git/tsfl/log/'     # 云端下发指令文件路径
send_dir = '/home/nvidia/git/tsfl/model/'   # 本地保存模型和日志的路径
client_name = 'jetson'

# 云端接收模型和日志的配置
remote_user = 'seeed'
remote_ip = '192.168.100.143'
remote_dir = '/home/seeed/fl/receive'

class_list = [
    'No entry', 'No passing', 'No vechiles',
    'output_20', 'output_30', 'output_40', 'output_50', 'output_70',
    'output_80', 'output_100', 'output_120',
    'Stop', 'Yield'
]
finished_rounds = set()

# ========== Preprocessing ==========
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

def load_data(data_path):
    images, labels = [], []
    class_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    for class_name in class_folders:
        if class_name not in class_list:
            continue
        class_index = class_list.index(class_name)
        img_dir = os.path.join(data_path, class_name)
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(preprocess(img))
                labels.append(class_index)
    return np.array(images), np.array(labels)

# ========== Custom loss ==========
def masked_categorical_crossentropy(class_mask):
    def loss_fn(y_true, y_pred):
        y_true_masked = y_true * class_mask
        y_pred_masked = y_pred * class_mask
        return tf.keras.losses.categorical_crossentropy(y_true_masked, y_pred_masked)
    return loss_fn

# ========== Model Architecture ==========
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

# ========== Training Function ==========
def run_training(round_id, data_path, global_model_path, output_model_path, output_log_path):
    print(f"[INFO] Starting training for round {round_id} using dataset at {data_path}")

    X, Y = load_data(data_path)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    Y = to_categorical(Y, num_classes=len(class_list))

    # Class mask based on local labels
    local_classes = np.unique(np.argmax(Y, axis=1)).tolist()
    class_mask = np.zeros(len(class_list), dtype=np.float32)
    for cls in local_classes:
        class_mask[cls] = 1.0
    class_mask = tf.constant(class_mask)

    model = create_model(len(class_list))

    if global_model_path and os.path.exists(global_model_path):
        print("[INFO] Loading global model weights...")
        model.load_weights(global_model_path)

    model.compile(optimizer='adam', loss=masked_categorical_crossentropy(class_mask), metrics=['accuracy'])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=42)
    model.fit(X_train, Y_train, validation_split=0.2, epochs=5, batch_size=32)
    loss, acc = model.evaluate(X_test, Y_test)

    print(f"[INFO] Finished training. Accuracy = {acc:.4f}")

    os.makedirs(send_dir, exist_ok=True)
    model.save(output_model_path)
    with open(output_log_path, 'w') as f:
        f.write(f"Client {client_name} finished round {round_id}, accuracy: {acc:.4f}\n")

    print("[INFO] Uploading model and log to server...")
    subprocess.run(['scp', output_model_path, f'{remote_user}@{remote_ip}:{remote_dir}/'])
    subprocess.run(['scp', output_log_path, f'{remote_user}@{remote_ip}:{remote_dir}/'])
    print("[INFO] Upload complete.")

# ========== Main Loop ==========
while True:
    log_files = [f for f in os.listdir(recv_dir) if f.endswith('.txt')]

    for log_file in log_files:
        log_path = os.path.join(recv_dir, log_file)
        with open(log_path, 'r') as f:
            content = f.read().strip()

        match = re.search(r'js(\d\d)', content)
        if not match:
            continue

        js_full = match.group(1)
        round_id = int(js_full[1])
        data_path = f'/home/nvidia/spd/js/js{js_full}'
        global_model_path = os.path.join(recv_dir, f'global_round{round_id - 1}.h5')
        output_model_path = os.path.join(send_dir, f'client_{client_name}_round{round_id}.h5')
        output_log_path = os.path.join(send_dir, f'client_{client_name}_round{round_id}.txt')

        if round_id in finished_rounds:
            continue

        if not os.path.exists(data_path):
            print(f"[WARN] Dataset not found: {data_path}")
            continue

        run_training(round_id, data_path, global_model_path, output_model_path, output_log_path)
        finished_rounds.add(round_id)

    time.sleep(10)
