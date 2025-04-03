from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from resources import *
import logging
import fitz 
import io
import csv
import json 

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
    extracted_skills_json = db.Column(db.String)

class JobPosting(db.Model):
    job_id = db.Column(db.String(13), primary_key=True)
    text = db.Column(db.Text, nullable=False)
    extracted_skills_json = db.Column(db.String)

with app.app_context():
    db.create_all()
    with open("extracted_job_skills.csv", "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            db.session.add(JobPosting(job_id=row["job_id"], text=row["cleaned_description"], extracted_skills_json=json.dumps(row["extracted_skills"])))
        file.close()
    db.session.commit()

# Function to add an entry
def add_entry(resumeText):
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

def combineSkillsToString(skills_json):
    finalString = ""
    for category, skillList in skills_json.items():
        if len(skillList) > 0:
            finalString += ((" ".join(skillList)) + " ")
    return finalString
    
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
        new_resume = Resume(name=name, text=extracted_text, extracted_skills_json="")
        db.session.add(new_resume)
        db.session.commit()

        return jsonify({'result': 'Resume successfully uploaded!'}), 200
    except Exception as e:
        app.logger.error(f"Error uploading resume: {str(e)}")
        return jsonify({'error': 'Failed to upload resume'}), 500

# Retrieve job postings
@app.route('/jobs', methods=['GET'])
def job():
    try:
        jobs = JobPosting.query.all()
        job_list = []
        for job in jobs:
            job_dict = {
                'job_id': job.job_id,
                'extracted_skills_json': json.loads(job.extracted_skills_json)
            }
            job_list.append(job_dict)
        return jsonify(job_list), 200
    except Exception as e:
        app.logger.error(f"Error retrieving jobs: {str(e)}")
        return jsonify({'error': 'Failed to retrieve jobs'}), 500    

# Match resume to job postings
@app.route('/match', methods=['GET'])
def match():
    try:
        name = request.args.get('name')
        pulledJobs = JobPosting.query.all()
        jobs = [(job.job_id, json.loads(json.loads(job.extracted_skills_json))) for job in pulledJobs]
        userSkills = json.loads(Resume.query.filter_by(name=name).first().extracted_skills_json)
        userCombinedSkills = combineSkillsToString(userSkills)
        app.logger.info(userCombinedSkills)
        jobsCombined = [(jobID, combineSkillsToString(skills)) for jobID, skills in jobs]
        app.logger.info(jobsCombined)
        matchingScores = find_similar_texts(userCombinedSkills, jobsCombined, matching_model)
        app.logger.info(matchingScores)
        return jsonify(matchingScores), 200
    except Exception as e:
        app.logger.error(f"Error matching: {str(e)}")
        return jsonify({'error': 'Failed to match'}), 500

@app.route('/extract_skills', methods=['GET'])
def extract_skills():
    try:
        name = request.args.get('name')
        if not name: 
            return jsonify({'error': 'No name provided'}), 500
        
        pulledResume = Resume.query.filter_by(name=name).first()
        if pulledResume is None:
            return jsonify({"error": "Resume not found"}), 404
        
        resume = pulledResume.text
        resume_ngrams = extract_ngrams_from_text(resume)  # Generate n-grams
        resume_skills = match_skills(resume_ngrams, resume)  # Match skills
        education_skills = resume_skills["Education Certification (EC)"]
        predictions = predict_entities(resume)
        outside_words = predictions["Outside (O)"]  # Extract "Outside" words/phrases from Model 2
        outside_text = preprocess_text(" ".join(outside_words))
        outside_ngrams = extract_ngrams_from_text(outside_text)  # Generate n-grams
        skills_from_outside = match_skills(outside_ngrams, outside_text)  # Match skills

        for category, skills in predictions.items():
            if category != "Outside (O)" and category != "Education Certification (EC)":
                for word in skills:
                    skills_from_outside[category].add(preprocess_text(word))
        
        skills_from_outside["Education Certification (EC)"] = education_skills
        outside_skills_dict = {key:list(itemList) for key,itemList in skills_from_outside.items()}

        pulledResume.extracted_skills_json = json.dumps(outside_skills_dict)  # Convert dict to string for storage
        db.session.commit()
        return jsonify(outside_skills_dict), 200
    except Exception as e:
        app.logger.error(f"Error extracting skills: {str(e)}")
        return jsonify({'error': 'Failed to extract skills'}), 500

if __name__ == '__main__':
    # Run the Flask app on all network interfaces
    app.run(host='0.0.0.0', port=5000)