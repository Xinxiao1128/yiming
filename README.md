# AI-Assisted Agriculture Pest and Disease Early Detection System
yiming - Sprint 1 (POC)

         Sprint 2 (ClearML Pipeline)

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


Features
1. Image Decompress
2. Preprocessing
3. CNN model-Based Disease Classification
4. Prediction Output and metrics
5. POC
6. Pipeline
7. CI/CD

# Model Evaluation Metrics
POC
1. Accuracy - 85.39%
2. Precision - 85.39%
3. Recall - 85.39%

Pipeline
4. Accuracy - 96.33%
5. Precision - 62.41%
6. Recall - 61.08%

![image](https://github.com/user-attachments/assets/6025ff74-717e-432d-8d91-66ef9b6ffbef)
![image](https://github.com/user-attachments/assets/05955bad-e0e3-4f6a-9edf-045ad69b5be7)
![image](https://github.com/user-attachments/assets/20ae9ac2-fb1f-4d6e-8f97-d69b670ffb34)


# Contributors:
1. AI Model Development : Jiawen Che, Xinxiao Li, Shiyuan Ruan
2. Data Preprocessing and Visualisation: Jiawen Che, Shiyuan Ruan, Xinxiao Li
3. Model Evaluation: Xinxiao Li, Jiawen Che
4. POC: Xinxiao Li, Jiawen Che
5. Pipeline (Level 1): Xinxiao Li, Shiyuan Ruan
6. Release note: Jiawen Che








