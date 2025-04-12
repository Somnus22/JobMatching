import re
import spacy
import sys
import subprocess
import nltk
from collections import defaultdict
import joblib
import gensim
import pandas as pd
import numpy as np
import os
import string
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

nltk.download("stopwords")

nlp = spacy.load('en_core_web_sm')
#Skill dictionaries
skill_dictionaries = {
    "Programming Language (PL)" : [
        "python","java",'javascript','java script',"c++","c","c#","ruby",'go','php','swift','typescript','type script','kotlin',
        'rust','scala','perl','r','matlab','assembly','visual basic','visual basic for applications','vba'
    ],

    "Framework (FW)": [
        "django", "flask", "spring", "spring framework", "spring boot", "react", "reactnative", "angular", "vue", "express", "fastapi", "asp.net", 
        ".net", "laravel", "ruby on rails", "symfony", "meteor", "gatsby", "svelte", "phoenix", "cake"
    ],

    "Database (DB)": [
        "datastore", "firestore", "metastore", "blob storage", "object storage", "file storage", "data storage", "disk storage", "cloud storage",
        "database", "knowledgebase", "firebase", "hbase", "database management", "database system", "database administration"
        "data warehouse", "data lake", "data mart", "data repository", "data center", "data server", "data modeling",
        "redis", "cassandra"
    ],

    "Cloud Platform (CP)":[
        "cloud", "cloud computing", "cloud storage", "cloud infrastructure", "cloud platform",
        "azure", "azure kubernetes service", "microsoft azure", 
        "gcp", "google cloud", "google cloud platform", "google cloud functions",
        "amazon web services", "ibm cloud", "oracle cloud", "digital ocean", "heroku",
        "serverless", "iaas", "paas", "saas", "faas", "baas", "caas",
        "s3", "ec2", "lambda", "elastic beanstalk", 
        "ecs", "eks", "fargate", "sqs", "sns", "rds", "redshift"
    ],

    "DevOps (DO)":[
        "devops", "devsecops",
        "ci", "ci/cd", "continuous integration", "continuous delivery", "continuous deployment",
        "continuous integration/continuous delivery", "continuous integration/continuous deployment",
        "version control", "git", "github", "gitlab",       
        "jenkins", "travisci", "circleci", "teamci", "bamboo",    
        "configuration management", "ansible", "puppet", "terraform",   
        "container", "containerization", "containerisation", 
        "docker", "dockerization", "dockerized", "dockerisation", "dockerised", "docker swarm", "kubernetes", "helm", 
        "monitor", "monitoring", "prometheus", "grafana",
        "unit testing", "integration testing", "selenium", "junit", "cypress", "jest",
        "load balancing", "scale", "scaling", "scalability", "disaster recovery", "chaos engineering"           
    ],

    "Network & Security (NS)":[
        "network", "network security", "network engineering", "network architecture",
        "security", "security analyst", "security engineering", "security architecture",
        "routing", "switching", "firewall", "firewall configuration", "vpn", "tcp", "tcp/ip",
        "dns", "dhcp", "ccna", "ccna/ccnp", "sd-wan", "vlans", "subnet",
        "risk", "incident response", "incident management", "identity and access management",
        "iam", "iams", "mfa", "multi-factor authentication", "vulnerability assessment",
        "vulnerability management", "vulnerability analysis",
        "crowdstrike", "carbon black", "cisco", "defense", "sso", "oauth", "saml", "hsm",
        "data encryption", "disk encryption", "owasp zap", "ethical hacking", "penetration testing"
    ],

    "Data Analysis & Science (DAS)":[
        "pandas", "numpy", "scikit learn", "tensorflow", "power bi", "excel", "tableau", "matplotlib",
        "data analytics", "data analysis", "data science", "data visualisation", "data visualization",'dashboard', "visualizing data",
        "ai", "artificial intelligence", "machine learning", "ml", "nlp", "natural language processing",
        "text analytics", "language model", "language models", "language modeling",
        "transaction management", "data transaction", "data transactions"
    ],

    "Software Engineering (SWE)":[
        "software development", "design patterns", "full stack", "fullstack", "full-stack",
        "code optimization", "code optimisation", "performance tuning", "code refactoring", "refactoring",
        "code review", "code reviews", "peer review", "peer reviews", 'test-driven'
        "microservice", "microservices", "MSA", "microservice architecture",
        "k8s", "eclipse", "eclipse ide", "hibernate", "hibernate orm", "jquery",
        "rest api", "rest apis", "restful api", "restful apis", "restful web service", "restful web services",
        "api development", "web service", "web services",
        "object oriented programming", "oop", "object oriented design", 
        "apache", "apache kafka", "apache tomcat", "apache maven", "apache ant", 
        "apache struts", "apache camel", "apache spark", "apache hadoop", "apache flink",
        "bootstrap", "front-end framework", "rabbitmq", "message queue", "message broker",
        "front-end", "frontend", "back-end", "backend", "web design", "ui", "user interface", "ux", "user experience", "ui/ux",
        "front-end development", "back-end development", "web development"
    ],

    "Project Management (PM)":[
        "agile", 'waterfall', 'atlassian', 'confluence', "jira", "trello", "asana", "kanban", "prince2", "stakeholder management", "scrum"
    ],

    "Education Certification (EC)":[
        "certified scrum master", "csm", "pmp", "aws certified", "azure certified", 
        "gcp certified", "cissp", "ccna", "ceh", "comptia", "cisa", "cism",
        "bachelor of", "bachelor of science", "bachelor of engineering", "bachelor of computer science",
        "bachelor of information technology", "bachelor of information systems", 
        "bachelor of cybersecurity", "bachelor of data science", "bachelor of software engineering",
        "B.S.", "BS", "B.E.", "BE", "B.C.S.", "BCS", "B.Tech.", "BTech",
        "Master's","master of science", "master of engineering", "master of computer science",
        "master of information technology", "master of information systems",
        "master of cybersecurity", "master of data science", "master of software engineering",
        "specialisation in software", "specialisation in data", "specialisation in cloud",
        "specialisation in security", "specialisation in networking", "specialisation in ai",
        "specialisation in machine learning", "specialization in software", "specialization in data",
        "specialization in cloud", "specialization in security", "specialization in networking",
        "specialization in ai", "specialization in machine learning",
        "minor in computer science", "minor in information technology", "minor in data science",
        "minor in software engineering", "major in computer science", "major in information technology",
        "major in data science", "major in software engineering", 
        "information systems","computer science", "computer engineering"
        "bachelor's degree", "bachelor degree", "ba/bs", "bs/ba", "b.s.", "b.a.", "undergraduate degree",
        "master's degree",
        "phd", "doctorate", "doctoral degree", "ph.d.",
        "computer science", "computer engineering", "software engineering", "information technology", 
        "information systems", "data science", "science"
    ],

    "Soft Skills (SS)":[
        "communication", "communication skill", "communication skills", 
        "verbal communication", "written communication",
        "leadership", "leadership skill", "leadership skills", 
        "decision making", "decision-making",
        "team building", "team-building",
        "mentoring", "coaching",
        "strategic thinking", "strategic-thinking",
        "teamwork", "collaboration", "team player",
        "interpersonal", "interpersonal skill", "interpersonal skills",
        "problem solving", "problem-solving",
        "critical thinking", "critical-thinking",
        "adaptability", "flexibility", "creativity",
        "self motivated", "self-motivated",
        "detail oriented", "detail-oriented",
        "work ethic",
        "time management", "time-management",
        "organizational", "organizational skill", "organizational skills",
        "multitasking", "prioritization",
        "project coordination",
        "conflict resolution", "conflict-resolution",
        "emotional intelligence",
        "presentation", "presentation skill", "presentation skills",
        "public speaking", "public-speaking",
        "negotiation", "negotiation skill", "negotiation skills",
        "customer service",
        "active listening", "active-listening",
        "stress management", "stress-management",
        "cultural awareness"
    ],
}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[0-9]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('.', '')
    return text

def preprocess_text_spacy(text):
    """Preprocess text using spaCy's capabilities"""
    text = text.lower()
    text = re.sub(r'[0-9]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Process the text with spaCy
    doc = nlp(text)
    
    return doc

def extract_ngrams_spacy(doc, max_n=4):
    """Extract n-grams using spaCy's token attributes"""
    tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
    
    all_ngrams = []
    for n in range(1, min(max_n + 1, len(tokens) + 1)):
        for i in range(len(tokens) - n + 1):
            all_ngrams.append(' '.join(tokens[i:i+n]))
    
    return all_ngrams

def extract_noun_phrases(doc):
    """Extract noun phrases which are likely to be skills"""
    return [chunk.text.lower() for chunk in doc.noun_chunks]

def extract_entities(doc):
    """Extract named entities which might represent technical skills"""
    return [ent.text.lower() for ent in doc.ents]

def sliding_window_match_spacy(doc, skill_dictionaries, window_size=6):
    """
    Extract skills where the words appear within a sliding window
    Using spaCy's token attributes for better matching
    """
    extracted_skills = defaultdict(set)
    
    # Convert tokens to text for window matching
    tokens = [token.text.lower() for token in doc 
              if not token.is_punct and not token.is_space]
    
    for category, skills in skill_dictionaries.items():
        for skill in skills:
            skill_words = skill.split()
            if len(skill_words) <= 1:
                continue
                
            for i in range(len(tokens) - window_size + 1):
                window = tokens[i:i+window_size]
                
                if all(word in window for word in skill_words):
                    try:
                        positions = [window.index(word) for word in skill_words]
                        scatter_factor = max(positions) - min(positions) + 1
                        
                        if scatter_factor <= len(skill_words) + 2:
                            clean_skill = skill.replace('.', '')
                            extracted_skills[category].add(clean_skill)
                    except ValueError:
                        continue
    
    return extracted_skills

def create_spacy_patterns(skill_dictionaries):
    """Create patterns for spaCy's Matcher"""
    patterns = []
    for category, skills in skill_dictionaries.items():
        for skill in skills:
            # Create pattern for exact match
            pattern = [{"LOWER": word} for word in skill.split()]
            patterns.append((skill, pattern))
    
    return patterns

def match_skills_spacy(text):
    """
    Match skills in text using spaCy's capabilities
    """
    extracted_skills = defaultdict(set)
    
    # Process text with spaCy
    doc = preprocess_text_spacy(text)
    
    # Extract n-grams
    ngrams = extract_ngrams_spacy(doc)
    
    # Extract noun phrases (potential skills)
    noun_phrases = extract_noun_phrases(doc)
    
    # Extract named entities
    entities = extract_entities(doc)
    
    # Combine all potential skill texts to check against dictionaries
    all_candidates = set(ngrams + noun_phrases + entities)
    
    # Custom regex patterns for specific technical terms
    js_frameworks = re.findall(r'\b\w+\.js\b', text.lower())
    sql_db = re.findall(r"\b((?:\w+)ql(?:\w+)?|(?:\w+)db)\b", text.lower())
    
    # Match against dictionaries
    for category, skills in skill_dictionaries.items():
        for skill in skills:
            # Direct match for single-word skills
            if ' ' not in skill and skill in all_candidates:
                clean_skill = skill.replace('.', '')
                extracted_skills[category].add(clean_skill)
            else:
                # For multi-word skills
                skill_lower = skill.lower()
                for candidate in all_candidates:
                    if skill_lower == candidate.lower():
                        clean_skill = skill.replace('.', '')
                        extracted_skills[category].add(clean_skill)
                        break
                    # Check if skill is contained within a longer phrase
                    elif len(candidate.split()) > len(skill.split()) and skill_lower in candidate.lower():
                        candidate_words = candidate.lower().split()
                        skill_words = skill_lower.split()
                        for i in range(len(candidate_words) - len(skill_words) + 1):
                            if ' '.join(candidate_words[i:i+len(skill_words)]) == skill_lower:
                                clean_skill = skill.replace('.', '')
                                extracted_skills[category].add(clean_skill)
                                break
    
    # Add sliding window matches
    sliding_matches = sliding_window_match_spacy(doc, skill_dictionaries)
    for category, skills in sliding_matches.items():
        extracted_skills[category].update(skills)
    
    # Add specialized matches from regex
    for skill in js_frameworks:
        clean_skill = skill.replace('.', '')
        extracted_skills["Framework (FW)"].add(clean_skill)
    
    for skill in sql_db:
        clean_skill = skill.replace('.', '')
        extracted_skills["Database (DB)"].add(clean_skill)
    
    # Add lemma-based matching for variations of the same word
    for token in doc:
        lemma = token.lemma_.lower()
        # Check if the lemma is in any skill list
        for category, skills in skill_dictionaries.items():
            for skill in skills:
                if ' ' not in skill and lemma == skill:
                    clean_skill = skill.replace('.', '')
                    extracted_skills[category].add(clean_skill)
    
    return extracted_skills

def word2features(sent, i):
    """Extract features for a given word in a sentence"""
    word = sent[i][0]  # The word itself
    
    # Basic features
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],  # Suffix
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }

# Features for words at the beginning of sentence
    if i == 0:
        features['BOS'] = True
    else:
        word_prev = sent[i-1][0]
        features.update({
            '-1:word.lower()': word_prev.lower(),
            '-1:word.istitle()': word_prev.istitle(),
            '-1:word.isupper()': word_prev.isupper(),
        })
    
    # Features for words at the end of sentence
    if i == len(sent)-1:
        features['EOS'] = True
    else:
        word_next = sent[i+1][0]
        features.update({
            '+1:word.lower()': word_next.lower(),
            '+1:word.istitle()': word_next.istitle(),
            '+1:word.isupper()': word_next.isupper(),
        })
    
    return features

def sent2features(sent):
    """Extract features for all words in a sentence"""
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    """Extract labels for all words in a sentence"""
    return [token[1] for token in sent]

crf = joblib.load('crf_model.joblib')
def predict_entities(text):
    # Tokenize the text (you might want to use your existing tokenization)
    tokens = text.split()  # Simple splitting, you might want something more sophisticated
    # Create features
    features = sent2features([(token, 'O') for token in tokens])  # Dummy labels
    
    # Predict
    predictions = crf.predict([features])[0]
    
    # Combine tokens with their predictions + generate dictionary format
    predictions_dict = generate_dictionary(list(zip(tokens, predictions)))
    
    return predictions_dict

def generate_dictionary(prediction_list):
    all_labels = {
        "PL": "Programming Languages",
        "FW": "Frameworks",
        "DB": "Databases",
        "CP": "Cloud Platforms",
        "DO": "DevOps",
        "NS": "Network & Security",
        "DAS": "Data Analysis & Science",
        "SWE": "Software Engineering",
        "PM": "Project Management",
        "EC": "Education Certification",
        "SS": "Soft Skills",
        "O": "Outside"
    }
    
    prediction_dict = {label: [] for label in all_labels}
    current_phrase = []
    current_label = None
    
    for word, label in prediction_list:
        if label == "O":
            if current_phrase and current_label:
                prediction_dict[current_label].append(" ".join(current_phrase))
                current_phrase = []
                current_label = None
            prediction_dict["O"].append(word)
        else:
            main_category = label[2:]
            if label.startswith("B-"):
                if current_phrase and current_label:
                    prediction_dict[current_label].append(" ".join(current_phrase))
                current_phrase = [word]
                current_label = main_category
            elif label.startswith("I-") and current_label == main_category:
                current_phrase.append(word)
    
    if current_phrase and current_label:
        prediction_dict[current_label].append(" ".join(current_phrase))
    
    normalized_dict = {
        "Programming Languages (PL)": [],
        "Frameworks (FW)": [],
        "Databases (DB)": [],
        "Cloud Platforms (CP)": [],
        "DevOps (DO)": [],
        "Network & Security (NS)": [],
        "Data Analysis & Science (DAS)": [],
        "Software Engineering (SWE)": [],
        "Project Management (PM)": [],
        "Education Certification (EC)": [],
        "Soft Skills (SS)": [],
        "Outside (O)": []
    }

    for key in prediction_dict.keys():
        new_key = all_labels[key] + " (" + key + ")"
        normalized_dict[new_key] = prediction_dict[key]
    
    return normalized_dict

def text2vec(text, model):
    words = text.split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return None
    
def find_similar_texts(input, corpus, model):
    input_vector = text2vec(input, model)
    if input_vector is None:
        return "Could not generate vector for input text"
    
    similarities = []
    for jobID, text in corpus:
        text_vector = text2vec(text, model)

        if text_vector is not None:
            similarity = cosine_similarity([input_vector], [text_vector])[0][0]
            similarities.append((jobID, similarity.item()))
        else:
            similarities.append((jobID, 0))

    top_similar = sorted(similarities, key=lambda x: x[1], reverse=True)

    return top_similar

matching_model =  Word2Vec.load("matching.model")

