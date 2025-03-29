# AI-Assisted Agriculture Pest and Disease Early Detection System
yiming - Sprint 1 (POC)

# Introduction

As the global population increases, the scale of agricultural planting expands, and pests and diseases pose an increasing threat to crops. Traditional pest and disease identification methods mainly rely on the experience of farmers and agricultural experts, and are not accurate enough. At the same time, it is often too late to discover the disease, and the cost of prevention and control is higher. Large-scale use of pesticides will also affect the edible safety of agricultural products. This project aims to develop an AI-assisted early identification system for agricultural pests and diseases, using computer vision and deep learning technology to improve the accuracy and timeliness of pest and disease identification, reduce crop losses, and reduce the abuse of pesticides.

The system will be used in the field of agricultural production and integrated into the agricultural management platform to provide farmers, agricultural experts and agricultural management agencies with intelligent pest and disease identification and prevention suggestions. The system uploads crop photos through mobile phones and other devices, uses AI models to analyze and identify the types of pests and diseases, and then combines agricultural databases to provide scientific and reasonable prevention and control plans.

According to the Food and Agriculture Organization of the United Nations (FAO), about 20% to 40% of the world's food crops are lost to pests and diseases each year, equivalent to an economic loss of US$220 billion. The accuracy of traditional manual identification of pests and diseases is between 60% and 75%, which is mainly limited by factors such as farmers' experience and on-site lighting conditions. The accuracy of pest and disease identification of this system is expected to reach 90% to 95%, significantly improving identification efficiency and accuracy. Due to the lack of accurate identification, the global application of pesticides has increased year by year, and the current annual average pesticide use is about 4 million tons. Through AI accurate identification and recommended prevention and control plans, 15% to 25% of unnecessary pesticide application can be reduced and the accuracy of pesticide application can be improved. Under the traditional pest and disease management model, farmers usually discover problems only after the disease spreads, resulting in a 10% to 25% reduction in crop yields. AI-assisted early detection can reduce losses to about 5%, effectively increasing crop yields and increasing farmers' income.

# Dataset: archive.zip 
https://drive.google.com/drive/u/1/folders/14ZGuTEeq0FnbUme3Ab_QS3IoTWs7_WrF 
The dataset contains 38 categories and consists of about 87K images of healthy and diseased crop leaves. The entire dataset was divided into training and validation sets in the ratio of 80/20 and the catalogue structure was retained. Subsequently a new catalogue containing 33 test images was created for prediction purpose.

# Code: Google colab 
https://colab.research.google.com/drive/1O_bYJLG3ydt2nmVM6Eu5zik10gxz__Fv?authuser =1
The current code is only used for POC after training using CNN models.
Features
1. Image Decompress
2. Preprocessing
3. CNN model-Based Disease Classification
4. Prediction Output and metrics
5. POC

![image](https://github.com/user-attachments/assets/6025ff74-717e-432d-8d91-66ef9b6ffbef)
![image](https://github.com/user-attachments/assets/3e43d309-3a04-4c1a-8334-01ca3a340ce1)
![image](https://github.com/user-attachments/assets/8a374e6b-f2df-4bf0-8b41-e438c7df6a66)
![image](https://github.com/user-attachments/assets/0b7056f1-f0c1-4fe8-a7c5-287a6dc45e07)
![image](https://github.com/user-attachments/assets/9367abaf-851c-438f-ac8a-a41de2f0ca90)
![image](https://github.com/user-attachments/assets/05955bad-e0e3-4f6a-9edf-045ad69b5be7)

Model Evaluation Metrics
Accuracy - 85.39%
Precision - 85.39%
Recall - 85.39%

Contributors
AI Model Development : Jiawen Che, Xinxiao Li, Shiyuan Ruan
Data Preprocessing and Visualisation: Jiawen Che, Shiyuan Ruan, Xinxiao Li
Model Evaluation: Xinxiao Li, Jiawen Che 
POC: Xinxiao Li, Jiawen Che








