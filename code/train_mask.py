import os
import re
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import subprocess

# ========== 配置区域 ==========
recv_dir = '/home/nvidia/git/tsfl/log/'  # 📥 接收日志和聚合模型的文件夹
client_name = 'jetson'              # 当前设备名称，用于命名输出文件
private_key_path = "/home/nvidia/.ssh/id_rsa"
class_list = [
    'No entry', 'No passing', 'No vechiles',
    'output_20', 'output_30', 'output_40', 'output_50', 'output_70',
    'output_80', 'output_100', 'output_120',
    'Stop', 'Yield'
]
output_model_path = f'client_{client_name}_round{{}}.h5'
output_log_path = f'client_{client_name}_round{{}}.txt'

# ========== 云端上传路径配置 ==========
server_ip = '192.168.100.143'  # ✅ 改为你的笔记本IP
server_user = 'seeed'        # ✅ 改为你的笔记本用户名
server_dir = '/home/seeed/fl/receive'  # ✅ 笔记本端的UPLOAD_DIR

# ========== 查找并解析日志 ==========
log_file = None
for f in os.listdir(recv_dir):
    if f.endswith('.txt'):
        log_file = os.path.join(recv_dir, f)
        break

if not log_file:
    raise FileNotFoundError("No instruction log found.")

with open(log_file, 'r') as f:
    log_content = f.read().strip()

match = re.search(r'js(\d+)', log_content)
if not match:
    raise ValueError("Dataset round (jsXX) not found in instruction log.")

match = re.search(r'js(\d\d)', log_content)  # 提取 js11, js12, js13
if not match:
    raise ValueError("Dataset round (jsXX) not found in instruction log.")

round_id = int(match.group(1)[1])  # ✅ 提取最后一位


data_path = f'/home/nvidia/spd/js/js{round_id}'

# ========== 检查聚合模型 ==========
agg_model_file = os.path.join(recv_dir, f'global_round{round_id - 1}.h5')
has_global_model = os.path.exists(agg_model_file)

# ========== 数据预处理 ==========
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

X, Y = load_data(data_path)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
Y = to_categorical(Y, num_classes=len(class_list))

# ========== 构建 masked loss ==========
local_classes = np.unique(np.argmax(Y, axis=1)).tolist()
class_mask = np.zeros(len(class_list), dtype=np.float32)
for cls in local_classes:
    class_mask[cls] = 1.0
class_mask = tf.constant(class_mask)

def masked_categorical_crossentropy(y_true, y_pred):
    y_true_masked = y_true * class_mask
    y_pred_masked = y_pred * class_mask
    return tf.keras.losses.categorical_crossentropy(y_true_masked, y_pred_masked)

# ========== 构建模型 ==========
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

model = create_model(num_classes=len(class_list))

if has_global_model:
    model.load_weights(agg_model_file)

model.compile(optimizer='adam', loss=masked_categorical_crossentropy, metrics=['accuracy'])

# ========== 训练 ==========
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=42)
model.fit(X_train, Y_train, validation_split=0.2, epochs=5, batch_size=32)
loss, acc = model.evaluate(X_test, Y_test)
print(f"Round {round_id} Accuracy: {acc:.4f}")

# ========== 保存结果 ==========
local_model = output_model_path.format(round_id)
local_log = output_log_path.format(round_id)

model.save(local_model)
with open(local_log, 'w') as f:
    f.write(f"Client {client_name} finished round {round_id}, accuracy: {acc:.4f}\n")

# ========== 上传到服务器 ==========
subprocess.run(['scp', '-i', private_key_path, local_model, f'{server_user}@{server_ip}:{server_dir}/'])
subprocess.run(['scp', '-i', private_key_path, local_log, f'{server_user}@{server_ip}:{server_dir}/'])
			

