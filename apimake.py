from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model('model.keras')

def preprocess_input(data):
    # 데이터를 NumPy 배열로 변환
    data = np.array(data).reshape(1, 28, 28, 1)  # 예: 28x28 이미지
    data = data / 255.0  # 정규화
    return data

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 요청 데이터 받기
        input_data = request.json.get('input')
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # 입력 데이터 전처리
        processed_data = preprocess_input(input_data)
        
        # 모델 예측
        prediction = model.predict(processed_data)
        
        # 결과 반환
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)