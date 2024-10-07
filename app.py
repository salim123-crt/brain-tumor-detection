import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///brain_tumor_detection.db'
db = SQLAlchemy(app)

# Définir la taille d'entrée
INPUT_SIZE = 64

# Charger le modèle pré-entraîné
model = load_model('brainTumor10EpochsCategorical.h5')

# S'assurer que le dossier d'upload existe
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(150), nullable=False)
    result = db.Column(db.String(50), nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)

def create_pdf(result, image_path, output_path):
    c = canvas.Canvas(output_path, pagesize=letter)
    c.drawString(100, 750, "Résultat de la Détection de Tumeur Cérébrale")
    c.drawString(100, 700, result)
    c.drawImage(image_path, 100, 450, width=400, height=300)
    c.save()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Traiter le fichier uploadé
            image = cv2.imread(file_path)
            img = Image.fromarray(image)
            img = img.resize((INPUT_SIZE, INPUT_SIZE))
            img = np.array(img)

            # Prétraiter l'image
            input_img = np.expand_dims(img, axis=0)

            # Obtenir les prédictions
            predictions = model.predict(input_img)
            predicted_class = np.argmax(predictions, axis=-1)

            if predicted_class == 0:
                message = "Aucune tumeur cérébrale détectée"
            else:
                message = "Une tumeur cérébrale a été détectée"

            # Enregistrer l'analyse dans la base de données
            with app.app_context():
                db.create_all()  # Assurez-vous d'être dans le contexte de l'application Flask
                analysis = Analysis(image_path=filename, result=message)
                db.session.add(analysis)
                db.session.commit()

            # Créer un rapport PDF
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'report_' + os.path.splitext(filename)[0] + '.pdf')
            create_pdf(message, file_path, pdf_path)

            return render_template('result.html', message=message, image_url=url_for('uploaded_file', filename=filename), pdf_url=url_for('uploaded_file', filename='report_' + os.path.splitext(filename)[0] + '.pdf'))
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(port=5001, debug=True)