# 🌐 Real-Time Face Mask Detection Using Deep Learning 🎭

## ✨ Overview
The COVID-19 pandemic underscored the importance of wearing masks as a preventive measure. This project focuses on building a **real-time face mask detection system** using **Deep Learning** and **Computer Vision** techniques. By leveraging **Convolutional Neural Networks (CNNs)** and **OpenCV**, we achieved an impressive **89% accuracy** 📊 in detecting whether a person is wearing a mask or not from real-time webcam footage.

## 🚀 Skills Demonstrated
This project enabled me to deepen my technical proficiency in several areas, including:

- 🧠 **Deep Learning & CNN Architecture**: Designed a **CNN** with multiple convolutional layers, batch normalization, and dropout to prevent overfitting.
- 🎥 **Computer Vision & OpenCV**: Used **Haar cascades** and **DNN-based face detection** for real-time facial recognition.
- 🔄 **Image Processing & Data Augmentation**: Applied **rotation, zoom, width/height shifts, and horizontal flips** to increase dataset diversity.
- 📊 **Model Evaluation & Optimization**: Fine-tuned hyperparameters using **Adam optimizer** and evaluated the model with **Precision, Recall, and F1-score**.
- 💻 **Python & TensorFlow**: Developed an end-to-end pipeline using **TensorFlow, Keras**, and **NumPy** for model training and inference.
- 🏗 **Modular Code Design**: Structured the project into reusable modules for data preprocessing, model training, and real-time detection.

## 📂 Dataset Details
The dataset used comprises **1,376 images**, categorized as:
- ✅ **690 images** of individuals wearing masks.
- ❌ **686 images** of individuals without masks.

We performed preprocessing steps such as **grayscale conversion, histogram equalization, and normalization** to improve model performance.

🔗 **Dataset Link:** [Face Mask Dataset](https://www.kaggle.com/datasets/vijaykumar1799/face-mask-detection)

## 🏗 Machine Learning Model & Techniques Used
We implemented a **deep CNN** architecture with the following specifications:

- **Input Shape**: 224x224x3 (RGB images resized for uniformity)
- **Layers**:
  - **Conv2D** layers with ReLU activation for feature extraction
  - **MaxPooling2D** layers to reduce spatial dimensions
  - **BatchNormalization** for stable learning
  - **Dropout (0.5)** to mitigate overfitting
  - **Flatten & Fully Connected Layers** for classification
- **Loss Function**: Binary Crossentropy (since it's a binary classification problem)
- **Optimizer**: Adam (Adaptive Moment Estimation) with **learning rate tuning**
- **Evaluation Metrics**: Accuracy, Precision, Recall, and F1-score

## 🔧 Use Cases & Applications
The **Face Mask Detection System** can be integrated into various real-world applications, such as:

- 🚪 **Automated Access Control**: Integrated with **automated doors** that only open when a person is wearing a mask, ensuring compliance in restricted areas.
- 🏢 **Smart Surveillance Systems**: Deployed in workplaces, hospitals, and public spaces to monitor mask compliance and enhance safety protocols.
- 🎥 **AI-Powered Security Cameras**: Implemented in **CCTV networks** for real-time monitoring and automated alerts.
- 🚆 **Public Transportation & Airports**: Used in buses, trains, and airports to reinforce mask-wearing regulations for passenger safety.

### 💡 Benefits
- ✅ **Reduces Manual Monitoring Effort**: Automates mask detection, minimizing the need for human intervention.
- 🎯 **Enhances Public Safety**: Ensures compliance in high-risk areas.
- ⚡ **Fast & Efficient**: Processes real-time footage with minimal latency.
- 🛠 **Scalable & Customizable**: Easily adaptable to different environments and regulations.

## 🎯 Conclusion
This project provided hands-on experience in building **AI-driven real-time applications**. It enhanced my understanding of **deep learning, model optimization, image preprocessing, and real-time detection pipelines**. The knowledge gained from this project is applicable to broader fields like **biometric security, medical imaging, and automated surveillance**.

---
📬 **Let’s connect!** Feel free to reach out for collaboration or discussion. 👨‍💻
