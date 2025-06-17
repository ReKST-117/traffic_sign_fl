import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === 1. 配置参数 ===
pretrained_model_path = "/home/nvidia/git/tsfl/premd.h5"
new_data_path = "/home/nvidia/spd/js11"  # 你可以换成 js12, rp11 等
image_size = (32, 32)
batch_size = 32
epochs = 10
freeze_pretrained = True  # 是否冻结除最后一层外的层

# === 2. 加载预训练模型 ===
model = load_model(pretrained_model_path)
print("✅ Pretrained model loaded.")

# === 3. 可选冻结除最后一层以外的所有层 ===
if freeze_pretrained:
    for layer in model.layers[:-1]:
        layer.trainable = False
    print("🧊 Pretrained layers frozen.")

# === 4. 编译模型（必须在更改 trainable 后重新编译）===
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === 5. 数据增强器和生成器 ===
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    new_data_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    new_data_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# === 6. 训练模型 ===
model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen
)

# === 7. 保存模型（覆盖或另存为新版本）===
model.save("/home/nvidia/git/tsfl/model.h5")  # 或保存为新版本：OtherNet_v2.h5
print("✅ Training complete. Model saved.")
