# AI-Assisted Agriculture Pest and Disease Early Detection System
yiming - Jiawen Che, Xinxiao Li, Shiyuan Ruan

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
https://colab.research.google.com/drive/1O_bYJLG3ydt2nmVM6Eu5zik10gxz__Fv?usp=sharing

The current code is only used for POC after training using CNN models.

https://colab.research.google.com/drive/1gMclg0HjoW9470ZN7LSvT79-skTcMsc0?usp=sharing
https://colab.research.google.com/drive/1Yk-JqXZ7cQddjH5iRJvRO6fU1gszH192?usp=sharing

These two copies of the code are for manual deployment of the ClearML Pipeline only.

https://drive.google.com/file/d/1lnjnZl_4YToYaQ9jGbPQhlfZ2R80mS1z/view?usp=sharing
https://colab.research.google.com/drive/1Qu5NwCSDJRRKLEWnWeThLjcWOOyaC0GG?usp=sharing
https://colab.research.google.com/drive/1iNHRLfJec6f6ZEn33Zx0-Brv_ph_uRKq?usp=sharing
https://colab.research.google.com/drive/1_0o3TJF3hx2kAT7gaRs_CuhFLABNNfqJ?usp=sharing

After conducting multiple experiments, we selected a suitable one for final submission.

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

# Best Model (After conducting multiple experiments, we finally selected a relatively stable and accurate algorithm with good accuracy and other parameters)
1. Best Accuracy - 95.31%
2. Batch Aize - 64
3. Dropout Rate - 0.5
4. Learning Rate - 0.0001
5. Weight Decay - 0.0001
6. Epochs - 10


![image](https://github.com/user-attachments/assets/6025ff74-717e-432d-8d91-66ef9b6ffbef)
![image](https://github.com/user-attachments/assets/05955bad-e0e3-4f6a-9edf-045ad69b5be7)
![image](https://github.com/user-attachments/assets/20ae9ac2-fb1f-4d6e-8f97-d69b670ffb34)
![image](https://github.com/user-attachments/assets/c67a35e5-b4f0-45d8-9068-1ecd56e93a04)
<img width="1217" alt="image" src="https://github.com/user-attachments/assets/a1390f0f-1687-435f-b86a-3f0dea69a0e8" />
<img width="1219" alt="image" src="https://github.com/user-attachments/assets/d794a2db-13eb-48ee-8fcb-c7e441d5dfb1" />
<img width="1214" alt="image" src="https://github.com/user-attachments/assets/e5d569ab-f397-4bbc-bb61-3dfe17d4f7eb" />


# GUI
<img width="176" alt="1748177453716" src="https://github.com/user-attachments/assets/8318823b-2cb5-46fc-9928-ec6b3415b543" />
<img width="173" alt="1748177463934" src="https://github.com/user-attachments/assets/917f46d0-132f-46f7-8cbc-90f59f5177f9" />
<img width="181" alt="1748177896850" src="https://github.com/user-attachments/assets/f28560ea-a71d-4903-b2f6-6e0c220ba89b" />
<img width="179" alt="1748177924870" src="https://github.com/user-attachments/assets/6c1fc2df-7979-436b-84da-abdd0f37926b" />
<img width="178" alt="1748177945557" src="https://github.com/user-attachments/assets/09fc4793-e842-4170-a064-d23649564cc0" />
<img width="173" alt="1748177965963" src="https://github.com/user-attachments/assets/b5daedb9-e9d0-4267-b32f-fc2418a89594" />
<img width="182" alt="1748177860323" src="https://github.com/user-attachments/assets/72e7b106-0681-4b3a-9913-3928a79207ad" />

# Contributors:
The entire assignment was divided fairly among the three of us, and the atmosphere within the group was good, with everyone working together smoothly.
1. AI Model Development : Jiawen Che, Xinxiao Li, Shiyuan Ruan
2. Data Preprocessing and Visualisation: Jiawen Che, Shiyuan Ruan, Xinxiao Li
3. Model Evaluation: Xinxiao Li, Jiawen Che
4. POC: Xinxiao Li, Jiawen Che
5. Pipeline (Level 1): Xinxiao Li, Shiyuan Ruan
6. Release note: Jiawen Che
7. CI/CD：Xinxiao Li
8. Pipeline (Level 2): Xinxiao Li, Shiyuan Ruan
9. Release note: Xinxiao Li, Jiawen Che
10. Presentation and Oral defence - Jiawen Che, Xinxiao Li, Shiyuan Ruan
