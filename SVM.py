import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
from PIL import Image

# 加载模型
model = load(r'C:\Users\armen\Downloads\EC-utbildning\2024-V.7-Machine learning\Xiaowen_Chen inlämningsuppgifter\mnist_SVM.joblib')


# 创建标题和描述
st.title('MNIST Digit Recognizer')
st.write('This application predicts the digit in an image.')

# Create a file uploader widget that accepts image files # 上传图片的功能
uploaded_file = st.file_uploader("Please upload a handwritten digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 读取文件并转换为PIL图像
    image = Image.open(uploaded_file).convert('L') # 转换为灰度图像
    
    # 处理图像以匹配模型预期的输入格式
    # 这可能包括调整大小、规范化和展平图像
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