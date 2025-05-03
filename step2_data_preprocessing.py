# ✅ Step 2: 解压 pest_dataset.zip + 图像预处理 + 上传至 ClearML

from clearml import Task, Dataset
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from glob import glob
from PIL import Image
import zipfile

# ==== 初始化 ClearML 任务 ====
task = Task.init(
    project_name='Pest Classification',
    task_name='Step 2 - Preprocessing'
)
task.execute_remotely(queue_name='default')  # ✅ 提交到远程代理

# ==== 参数配置 ====
params = task.connect({
    'dataset_task_id': '',  # ⚠️ 运行前填写 Step 1 返回的 dataset_id
    'image_size': 256,
    'test_size': 0.2,
    'random_state': 42
})

# ==== 步骤 2.1：下载第 1 步上传的 ZIP 数据集 ====
dataset = Dataset.get(dataset_id=params['dataset_task_id'])
local_path = dataset.get_local_copy()

# ==== 步骤 2.2：解压 ZIP ====
zip_path = os.path.join(local_path, 'pest_dataset.zip')  # zip 文件名要与实际一致
extract_dir = os.path.join(local_path, 'unzipped')

os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"✅ 解压完成，目录：{extract_dir}")

# ==== 步骤 2.3：图像预处理 ====
img_size = (params['image_size'], params['image_size'])
x_data, y_data = [], []

# 建立类别标签映射
dirs = sorted([d for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))])
label2id = {name: idx for idx, name in enumerate(dirs)}

for class_name in dirs:
    class_dir = os.path.join(extract_dir, class_name)
    for img_file in glob(os.path.join(class_dir, '*.jpg')):
        try:
            img = Image.open(img_file).convert('RGB').resize(img_size)
            x_data.append(np.array(img))
            y_data.append(label2id[class_name])
        except Exception as e:
            print(f"⚠️ 图像处理失败：{img_file}, 错误: {e}")
            continue

x_data = np.array(x_data)
y_data = np.array(y_data)

# ==== 步骤 2.4：划分训练和测试集 ====
X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=params['test_size'], random_state=params['random_state'])

# ==== 步骤 2.5：保存并上传为 ClearML 数据集 ====
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
np.save('label2id.npy', label2id)

output_dataset = Dataset.create(dataset_name='Pest Dataset Preprocessed', dataset_project='Pest Classification')
output_dataset.add_files('X_train.npy')
output_dataset.add_files('X_test.npy')
output_dataset.add_files('y_train.npy')
output_dataset.add_files('y_test.npy')
output_dataset.add_files('label2id.npy')
output_dataset.upload()
output_dataset.finalize()

print('✅ 图像预处理完成，预处理数据已上传 ClearML')
