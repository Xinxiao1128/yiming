# AI-Assisted Agriculture Pest and Disease Early Detection System
yiming

Sprint 1 (POC)

Sprint 2 (ClearML Pipeline)

Sprint 3 (ClearML CI/CD、HPO、GUI)

# Introduction

As the global population increases, the scale of agricultural planting expands, and pests and diseases pose an increasing threat to crops. Traditional pest and disease identification methods mainly rely on the experience of farmers and agricultural experts, and are not accurate enough. At the same time, it is often too late to discover the disease, and the cost of prevention and control is higher. Large-scale use of pesticides will also affect the edible safety of agricultural products. This project aims to develop an AI-assisted early identification system for agricultural pests and diseases, using computer vision and deep learning technology to improve the accuracy and timeliness of pest and disease identification, reduce crop losses, and reduce the abuse of pesticides.

The system will be used in the field of agricultural production and integrated into the agricultural management platform to provide farmers, agricultural experts and agricultural management agencies with intelligent pest and disease identification and prevention suggestions. The system uploads crop photos through mobile phones and other devices, uses AI models to analyze and identify the types of pests and diseases, and then combines agricultural databases to provide scientific and reasonable prevention and control plans.

According to the Food and Agriculture Organization of the United Nations (FAO), about 20% to 40% of the world's food crops are lost to pests and diseases each year, equivalent to an economic loss of US$220 billion. The accuracy of traditional manual identification of pests and diseases is between 60% and 75%, which is mainly limited by factors such as farmers' experience and on-site lighting conditions. The accuracy of pest and disease identification of this system is expected to reach 90% to 95%, significantly improving identification efficiency and accuracy. Due to the lack of accurate identification, the global application of pesticides has increased year by year, and the current annual average pesticide use is about 4 million tons. Through AI accurate identification and recommended prevention and control plans, 15% to 25% of unnecessary pesticide application can be reduced and the accuracy of pesticide application can be improved. Under the traditional pest and disease management model, farmers usually discover problems only after the disease spreads, resulting in a 10% to 25% reduction in crop yields. AI-assisted early detection can reduce losses to about 5%, effectively increasing crop yields and increasing farmers' income.

# Dataset: archive.zip 
https://drive.google.com/drive/u/1/folders/14ZGuTEeq0FnbUme3Ab_QS3IoTWs7_WrF 
The dataset contains 38 categories and consists of about 87K images of healthy and diseased crop leaves. The entire dataset was divided into training and validation sets in the ratio of 80/20 and the catalogue structure was retained. Subsequently a new catalogue containing 33 test images was created for prediction purpose.
The second part of the dataset has been uploaded into clearml.

# Code: Google colab 
https://colab.research.google.com/drive/1O_bYJLG3ydt2nmVM6Eu5zik10gxz__Fv?authuser=1
The current code is only used for POC after training using CNN models.

https://colab.research.google.com/drive/1gMclg0HjoW9470ZN7LSvT79-skTcMsc0?usp=sharing
https://colab.research.google.com/drive/1Yk-JqXZ7cQddjH5iRJvRO6fU1gszH192?usp=sharing
These two copies of the code are for manual deployment of the ClearML Pipeline only.

https://colab.research.google.com/drive/1Qu5NwCSDJRRKLEWnWeThLjcWOOyaC0GG?usp=sharing
https://colab.research.google.com/drive/1Yk-JqXZ7cQddjH5iRJvRO6fU1gszH192?usp=sharing

# Features
1. Image Decompress
2. Preprocessing
3. CNN model-Based Disease Classification
4. Prediction Output and metrics
5. POC
6. Pipeline level 1
7. CI/CD
8. Pipeline level 2 (HPO+Final)

# Model Evaluation Metrics
POC (1-3)/Pipeline (4-6)
1. Accuracy - 85.39%
2. Precision - 85.39%
3. Recall - 85.39%
4. Accuracy - 96.33%
5. Precision - 62.41%
6. Recall - 61.08%


![image](https://github.com/user-attachments/assets/6025ff74-717e-432d-8d91-66ef9b6ffbef)
![image](https://github.com/user-attachments/assets/05955bad-e0e3-4f6a-9edf-045ad69b5be7)
![image](https://github.com/user-attachments/assets/20ae9ac2-fb1f-4d6e-8f97-d69b670ffb34)
![image](https://github.com/user-attachments/assets/c67a35e5-b4f0-45d8-9068-1ecd56e93a04)

# GUI
<img width="176" alt="1748177453716" src="https://github.com/user-attachments/assets/8318823b-2cb5-46fc-9928-ec6b3415b543" />
<img width="173" alt="1748177463934" src="https://github.com/user-attachments/assets/917f46d0-132f-46f7-8cbc-90f59f5177f9" />
![be576f3d84876555bfe5c2c7f77d121](https://github.com/user-attachments/assets/5a05d210-327f-4e8c-9cdc-45f44b079357)
![2b19eeacb4df4ab8506bd5ad5f6a8cd](https://github.com/user-attachments/assets/fc6957a9-722a-40ef-890b-2f2563257f7d)
![a9ed5441f58769ce98abf7022a9eb83](https://github.com/user-attachments/assets/c3a4c57c-5741-49fa-b633-7eda965aaf83)
![9c16cbeb581dd8fd0ecb71625b51f21](https://github.com/user-attachments/assets/b926d855-c2db-4ed8-9f15-f5bb79e79260)
![679b19bfd6b6210e9f4ff01ef4c7fbf](https://github.com/user-attachments/assets/f2128c50-3d06-44e4-8428-0c063bd8380b)

# Contributors:
1. AI Model Development : Jiawen Che, Xinxiao Li, Shiyuan Ruan
2. Data Preprocessing and Visualisation: Jiawen Che, Shiyuan Ruan, Xinxiao Li
3. Model Evaluation: Xinxiao Li, Jiawen Che
4. POC: Xinxiao Li, Jiawen Che
5. Pipeline (Level 1): Xinxiao Li, Shiyuan Ruan
6. Release note: Jiawen Che
7. CI/CD：Xinxiao Li
8. Pipeline (Level 2): Xinxiao Li, Shiyuan Ruan
9. Release note: Xinxiao Li, Jiawen Che
