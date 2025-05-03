# ✅ Step 1：自动上传大型压缩数据集到 ClearML Dataset（适用于 Google Colab）

from clearml import Task, Dataset
import os
import shutil

# ==== (1) 初始化 ClearML 任务 ====
task = Task.init(
    project_name='Agri-Pest-Detection',
    task_name='Step 1 - Upload archive.zip (Colab)'
)
task.execute_remotely(queue_name='default')  # ✅ 推送至远程 agent 执行

# ==== (2) 挂载 Google Drive ====
from google.colab import drive
drive.mount('/content/drive')

# ==== (3) 拷贝数据集压缩包到当前目录 ====
# ⚠️ 请根据你在 Google Drive 中实际路径修改下面这一行路径：
drive_zip_path = '/content/drive/MyDrive/42174_AI_Studio-yiming/archive.zip'
local_zip_path = './archive.zip'

# 如果文件不存在，则从 Drive 拷贝
if not os.path.exists(local_zip_path):
    shutil.copy(drive_zip_path, local_zip_path)

# 确认数据存在
assert os.path.exists(local_zip_path), "❌ archive.zip 未找到，请确认 Google Drive 路径是否正确。"

# ==== (4) 上传为 ClearML Dataset ====
dataset = Dataset.create(
    dataset_name='PlantVillage-Zip',
    dataset_project='Agri-Pest-Detection'
)
dataset.add_files(path=local_zip_path)
dataset.upload()
dataset.finalize()

print("✅ archive.zip 上传成功，已注册为 ClearML Dataset。")
