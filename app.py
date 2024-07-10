from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Charger le modèle
model = load_model('model.h5')

# Définir une route pour l'API
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Lire l'image
    img = Image.open(io.BytesIO(file.read()))
    img = img.resize((150, 150))  # Assurez-vous que la taille correspond à celle attendue par le modèle
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Prédiction
    prediction = model.predict(img_array)
    result = np.argmax(prediction, axis=1)[0]  # ou utilisez votre propre logique pour décoder les prédictions

    return jsonify({'prediction': int(result)})

if __name__ == '__main__':
    app.run(debug=True)
