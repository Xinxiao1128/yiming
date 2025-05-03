# 📦 ClearML Pipeline: Step 3 - CNN Model Training and Artifact Logging
from clearml import Task, Dataset
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# ==== (1) Initialize ClearML task ====
task = Task.init(project_name='Agri-Pest-Detection', task_name='Step 3 - Model Training')
task.execute_remotely(queue_name='default')  # ✅ Submit to remote ClearML agent

# ==== (2) Set task parameters ====
params = task.connect({
    'dataset_task_id': '',  # ⚠️ Manually insert the Dataset task ID from Step 2
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 0.0001
})

# ==== (3) Load preprocessed dataset ====
dataset = Dataset.get(task_id=params['dataset_task_id'])
local_path = dataset.get_local_copy()

X_train = np.load(f"{local_path}/X_train.npy")
X_test = np.load(f"{local_path}/X_test.npy")
y_train = np.load(f"{local_path}/y_train.npy")
y_test = np.load(f"{local_path}/y_test.npy")
label2id = np.load(f"{local_path}/label2id.npy", allow_pickle=True).item()

# Optional: Normalize pixel values to [0, 1]
# X_train, X_test = X_train / 255.0, X_test / 255.0

# ==== (4) Build CNN model ====
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(256, 256, 3)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(label2id), activation='softmax')
])

# ==== (5) Compile and train model ====
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=params['epochs'],
    batch_size=params['batch_size']
)

# ==== (6) Save model and training curves ====
model.save('cnn_model.h5')
task.upload_artifact("model", artifact_object='cnn_model.h5')

# Accuracy plot
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("accuracy_curve.png")
task.upload_artifact("accuracy_curve", artifact_object='accuracy_curve.png')

# Loss plot
plt.clf()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss_curve.png")
task.upload_artifact("loss_curve", artifact_object='loss_curve.png')

print("✅ Model training complete and all artifacts uploaded to ClearML.")
