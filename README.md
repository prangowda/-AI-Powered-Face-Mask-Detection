# 😷 AI-Powered Face Mask Detection  

## 📌 Overview  
This AI-powered Face Mask Detection tool detects whether a person is wearing a mask in real-time using **Deep Learning and Computer Vision**. It uses a **Convolutional Neural Network (CNN)** trained on a dataset of masked and unmasked faces. The system works with **live webcam feeds** and can be deployed for **real-time CCTV monitoring**.  

## 🛠️ Technologies Used  
| **Technology** | **Purpose** |
|--------------|------------|
| **Python** | Programming Language |
| **OpenCV (cv2)** | Real-time video processing |
| **TensorFlow/Keras** | Deep Learning framework |
| **Matplotlib** | Data visualization |
| **NumPy** | Array processing for images |

---

## 📜 Features  
✅ **Real-time mask detection via webcam**  
✅ **Trained CNN model for high accuracy**  
✅ **Can be deployed on CCTV cameras**  
✅ **Fast and lightweight detection**  
✅ **Alerts for non-compliance**  

---

## 🚀 Installation & Setup  

### **1️⃣ Install Dependencies**  
Run the following command to install required libraries:  
```sh
pip install opencv-python numpy tensorflow keras matplotlib
```

### **2️⃣ Download Dataset**  
You need a dataset containing images of **people with and without masks**. You can use the **Face Mask Dataset** from Kaggle:  
🔗 [Download Face Mask Dataset](https://www.kaggle.com/andrewmvd/face-mask-detection)  

Move the dataset into a folder named `dataset/` inside your project directory.

---

## 📂 Project Structure  
```
/AI_Face_Mask_Detection
│── dataset/                     # Training dataset
│── face_mask_detector.py        # Main detection script
│── train_mask_detector.py       # Model training script
│── mask_detector.model          # Saved trained model
│── README.md                    # Documentation
```

---

## 🎯 How to Run the Project  

### **1️⃣ Train the CNN Model**  
Run the following command to train the model:  
```sh
python train_mask_detector.py
```
Once trained, the model will be saved as `mask_detector.model`.

### **2️⃣ Run Real-Time Mask Detection**  
Execute the following command to start real-time face mask detection:  
```sh
python face_mask_detector.py
```
The webcam will turn on, and the system will **detect if a person is wearing a mask or not**.

---

## 📂 Example Output  

| **Image** | **Prediction** |
|-----------|--------------|
| ✅ Mask Detected | ❌ No Mask Detected |

---

## 🚀 Future Enhancements  
- ✅ **Deploy on CCTV for real-time monitoring**  
- ✅ **Add sound alerts for non-mask violations**  
- ✅ **Store logs in cloud for reporting**  
- ✅ **Enhance detection accuracy with a larger dataset**  
