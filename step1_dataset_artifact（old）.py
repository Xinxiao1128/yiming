# ✅ Step 1: Automatically upload large zip dataset to ClearML Dataset (for Google Colab)

from clearml import Task, Dataset
import os
import shutil

# ==== (1) Initialize ClearML Task ====
task = Task.init(
    project_name='Agri-Pest-Detection',
    task_name='Step 1 - Upload archive.zip (Colab)'
)
task.execute_remotely(queue_name='default')  # ✅ Submit to remote ClearML agent

# ==== (2) Mount Google Drive ====
from google.colab import drive
drive.mount('/content/drive')

# ==== (3) Copy archive.zip to current directory ====
# ⚠️ Modify the following path according to your actual Google Drive location:
drive_zip_path = '/content/drive/MyDrive/42174_AI_Studio-yiming/archive.zip'
local_zip_path = './archive.zip'

# Copy the zip file if it doesn't already exist locally
if not os.path.exists(local_zip_path):
    shutil.copy(drive_zip_path, local_zip_path)

# Ensure the zip file exists
assert os.path.exists(local_zip_path), "❌ archive.zip not found. Please check the Google Drive path."

# ==== (4) Upload as ClearML Dataset ====
dataset = Dataset.create(
    dataset_name='PlantVillage-Zip',
    dataset_project='Agri-Pest-Detection'
)
dataset.add_files(path=local_zip_path)
dataset.upload()
dataset.finalize()

print("✅ archive.zip uploaded successfully and registered as a ClearML Dataset.")
