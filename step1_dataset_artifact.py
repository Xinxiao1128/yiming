from clearml import Task, Dataset
import os
import shutil

# ==== (1) 初始化 ClearML 任务 ====
task = Task.init(
    project_name='Pest Classification',
    task_name='Step 1 - Upload pest ZIP from Google Drive'
)
task.execute_remotely(queue_name='default')  # ✅ 提交给远程执行

# ==== (2) 挂载 Google Drive ====
from google.colab import drive
drive.mount('/content/drive')

# ==== (3) 指定你 ZIP 的位置 ====
# ⚠️ 请根据你自己的 Google Drive 路径修改这里
drive_zip_path = '/content/drive/MyDrive/pest_dataset.zip'
local_zip_path = './pest_dataset.zip'

# 拷贝压缩包到当前目录
if not os.path.exists(local_zip_path):
    shutil.copy(drive_zip_path, local_zip_path)

assert os.path.exists(local_zip_path), "❌ ZIP 文件不存在，请确认路径正确。"

# ==== (4) 上传为 ClearML Dataset ====
dataset = Dataset.create(
    dataset_name='Pest Dataset ZIP',
    dataset_project='Pest Classification'
)
dataset.add_files(path=local_zip_path)
dataset.upload()
dataset.finalize()

print("✅ pest_dataset.zip 上传成功，dataset_id:", dataset.id)
.")
