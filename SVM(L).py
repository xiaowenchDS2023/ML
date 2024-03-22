import streamlit as st
from joblib import load
import numpy as np
from PIL import Image

# 定义加载模型的函数
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = "mnist_SVM.joblib"  # 模型文件相对路径
    model = load(model_path)
    return model

# 创建标题和描述
st.title('MNIST Digit Recognizer')
st.write('This application predicts the digit in an image.')

# 加载模型
model = load_model()

# 上传图片的功能
uploaded_file = st.file_uploader("Please upload a handwritten digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 读取文件并转换为PIL图像
    image = Image.open(uploaded_file).convert('L') # 转换为灰度图像
    
    # 处理图像以匹配模型预期的输入格式
    processed_image = np.array(image.resize((28, 28), Image.Resampling.LANCZOS))
    processed_image = processed_image / 255.0  # 规范化像素值
    processed_image = processed_image.flatten().reshape(1, -1)  # 展平图像为一维数组

    # 显示图片
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # 进行预测
    st.write("Predicting...")
    prediction = model.predict(processed_image)
    st.write('Predicted Digit:', prediction[0])

import os
from joblib import load

# 定义加载模型的函数
def load_model():
    model_path = "mnist_SVM.joblib"  # 模型文件相对路径
    print("Model path:", model_path)
    print("Model file exists:", os.path.exists(model_path))  # 检查模型文件是否存在
    model = load(model_path)
    return model

# 加载模型
try:
    model = load_model()
except Exception as e:
    print("Error loading model:", e)
