from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from resources import *
import logging
import fitz 
import io

app = Flask(__name__)
CORS(app)  # Allow all origins
logging.basicConfig(level=logging.INFO)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define a simple database with a string field
class Resume(db.Model):
    name = db.Column(db.String(255), primary_key=True)
    text = db.Column(db.Text, nullable=False)

# Create the database
with app.app_context():
    db.create_all()

# Function to add an entry
def add_entry(resumeText):
    with app.app_context():  # Required to access the database in a Flask app
        new_entry = Resume(text=resumeText)
        db.session.add(new_entry)
        db.session.commit()

def extract_text_from_pdf(file):
    """Extract text from a PDF file directly from file stream using PyMuPDF."""
    pdf_data = file.read()  # Read the file data
    file.seek(0)
    doc = fitz.open(stream=io.BytesIO(pdf_data), filetype="pdf")  # Open PDF from stream
    text = "\n".join([page.get_text() for page in doc])  # Extract text from each page
    return text

@app.route('/uploadResume', methods=['POST'])
def uploadResume():
    try:
        # Check if file and name are provided in the request
        if 'file' not in request.files or 'name' not in request.form:
            return jsonify({'error': 'No file or name provided'}), 400
        
        file = request.files['file']
        name = request.form['name']  # Retrieve the name from the form

        # Extract text from the PDF
        extracted_text = extract_text_from_pdf(file)

        # Store the resume in the database
        new_resume = Resume(name=name, text=extracted_text)
        db.session.add(new_resume)
        db.session.commit()

        return jsonify({'result': 'Resume successfully uploaded!'}), 200
    except Exception as e:
        app.logger.error(f"Error uploading resume: {str(e)}")
        return jsonify({'error': 'Failed to upload resume'}), 500
    
# Apply model 1 
@app.route('/model1', methods=['GET'])
def model_1():
    try:
        name = request.args.get('name')
        if not name: 
            return jsonify({'error': 'No name provided'}), 500
        
        pulledResume = Resume.query.filter_by(name=name).first()
        if pulledResume is None:
            return jsonify({"error": "Resume not found"}), 404
        
        resume = pulledResume.text

        resume_ngrams = extract_ngrams_from_text(resume)
        resume_skills = match_skills(resume_ngrams, resume)

        resume_skills_dict = {key:list(itemList) for key,itemList in resume_skills.items()}
        return jsonify(resume_skills_dict), 200
    except Exception as e:
        app.logger.error(f"Error applying model 1 for task 1: {str(e)}")
        return jsonify({'error': 'Failed to apply model 1'}), 500

# @app.route('/model2', methods=['GET'])
# def model_2():
#     try:
#         s
#     except Exception as e:
#         app.logger.error(f"Error applying model 1 for task 1: {str(e)}")
#         return jsonify({'error': 'Failed to retrieve products'}), 500

if __name__ == '__main__':
    # Run the Flask app on all network interfaces
    app.run(host='0.0.0.0', port=5000)