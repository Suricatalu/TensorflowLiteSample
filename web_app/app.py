#!/usr/bin/env python3
"""
狗貓分類 Web 應用
使用 Flask 建立網頁介面，上傳圖片進行預測
"""

import os
import io
import base64
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # 實際使用時請更改

# 配置
UPLOAD_FOLDER = 'web_app/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MODEL_PATH = 'models/cat_dog_classifier'
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# 確保上傳資料夾存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 全域變數儲存模型
model = None

def load_model():
    """載入訓練好的模型"""
    global model
    if model is None:
        try:
            print(f"正在載入模型: {MODEL_PATH}")
            model = tf.keras.models.load_model(MODEL_PATH)
            print("模型載入成功！")
        except Exception as e:
            print(f"模型載入失敗: {e}")
            model = None
    return model

def allowed_file(filename):
    """檢查檔案類型是否允許"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image, target_size=(180, 180)):
    """預處理圖片"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize(target_size)
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_image(image):
    """預測圖片"""
    current_model = load_model()
    if current_model is None:
        return None, "模型未載入"
    
    try:
        # 預處理圖片
        img_array = preprocess_image(image)
        
        # 進行預測
        prediction = current_model.predict(img_array, verbose=0)
        confidence = prediction[0][0]
        
        # 判斷結果
        if confidence > 0.5:
            predicted_class = "Dog"
            probability = confidence
        else:
            predicted_class = "Cat"
            probability = 1 - confidence
        
        return {
            'class': predicted_class,
            'confidence': probability,
            'raw_prediction': float(confidence)
        }, None
        
    except Exception as e:
        return None, f"預測失敗: {str(e)}"

def image_to_base64(image):
    """將 PIL 圖片轉換為 base64 字串"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

@app.route('/')
def index():
    """首頁"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """處理圖片預測請求"""
    if 'file' not in request.files:
        flash('請選擇圖片檔案')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('請選擇圖片檔案')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            # 讀取圖片
            image = Image.open(file.stream)
            
            # 進行預測
            result, error = predict_image(image)
            
            if error:
                flash(error)
                return redirect(url_for('index'))
            
            # 將圖片轉換為 base64 用於顯示
            image_b64 = image_to_base64(image)
            
            return render_template('result.html', 
                                 result=result,
                                 image_data=image_b64,
                                 filename=file.filename)
            
        except Exception as e:
            flash(f'處理圖片時發生錯誤: {str(e)}')
            return redirect(url_for('index'))
    
    else:
        flash('不支援的檔案格式，請上傳 PNG, JPG, JPEG, GIF 或 BMP 檔案')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API 介面預測"""
    if 'file' not in request.files:
        return jsonify({'error': '請提供圖片檔案'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': '請選擇圖片檔案'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': '不支援的檔案格式'}), 400
    
    try:
        # 讀取圖片
        image = Image.open(file.stream)
        
        # 進行預測
        result, error = predict_image(image)
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify({
            'success': True,
            'prediction': result['class'],
            'confidence': f"{result['confidence']:.2%}",
            'raw_confidence': result['confidence'],
            'filename': file.filename
        })
        
    except Exception as e:
        return jsonify({'error': f'處理圖片時發生錯誤: {str(e)}'}), 500

@app.route('/health')
def health():
    """健康檢查端點"""
    current_model = load_model()
    return jsonify({
        'status': 'ok' if current_model is not None else 'error',
        'model_loaded': current_model is not None
    })

if __name__ == '__main__':
    print("正在啟動狗貓分類 Web 應用...")
    print(f"模型路徑: {MODEL_PATH}")
    
    # 載入模型
    load_model()
    
    print("應用啟動完成！")
    print("請開啟瀏覽器訪問: http://localhost:5050")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
