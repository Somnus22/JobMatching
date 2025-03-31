import re
import nltk
from nltk.util import ngrams
from collections import defaultdict
import joblib

nltk.download('punkt_tab')

skill_dictionaries = {
    "Programming Languages (PL)" : [
        "python","java",'javascript','java script',"c++","c","c#","ruby",'go','php','swift','typescript','type script','kotlin',
        'rust','scala','perl','r','matlab','assembly','visual basic','visual basic for applications','vba'
    ],

    "Frameworks (FW)": [
        "django", "flask", "spring", "spring framework", "spring boot", "react", "reactnative", "angular", "vue", "express", "fastapi", "asp.net", 
        ".net", "laravel", "ruby on rails", "symfony", "meteor", "gatsby", "svelte", "phoenix", "cake"
    ],

    "Databases (DB)": [
        "datastore", "firestore", "metastore", "blob storage", "object storage", "file storage", "data storage", "disk storage", "cloud storage",
        "database", "knowledgebase", "firebase", "hbase", "database management", "database system", "database administration"
        "data warehouse", "data lake", "data mart", "data repository", "data center", "data server", "data modeling",
        "redis", "cassandra"
    ],

    "Cloud Platforms (CP)":[
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

def extract_ngrams_from_text(text, max_n=4):
    processed_text = preprocess_text(text)
    tokens = nltk.word_tokenize(processed_text)
    
    all_ngrams = []
    for n in range(1, max_n+1):
        n_gram_list = list(ngrams(tokens, n))
        all_ngrams.extend([' '.join(gram) for gram in n_gram_list])
    
    return all_ngrams

def sliding_window_match(tokens, skill_dictionaries, window_size=6):
    """
    Extract skills where the words appear within a sliding window but not necessarily consecutively.
    """
    extracted_skills = defaultdict(set)
    
    for category, skills in skill_dictionaries.items():
        for skill in skills:
            skill_words = skill.split()
            if len(skill_words) <= 1:
                continue
                
            for i in range(len(tokens) - window_size + 1):
                window = tokens[i:i+window_size]
                
                window_lower = [w.lower() for w in window]
                
                if all(word in window_lower for word in skill_words):
                    try:
                        positions = [window_lower.index(word) for word in skill_words]
                        scatter_factor = max(positions) - min(positions) + 1
                        
                        if scatter_factor <= len(skill_words) + 2:
                            clean_skill = skill.replace('.', '')
                            extracted_skills[category].add(clean_skill)
                    except ValueError:
                        continue
    
    return extracted_skills
  
    
def match_skills(ngram_list, text):
    extracted_skills = defaultdict(set)
    
    processed_text = preprocess_text(text)
    tokens = nltk.word_tokenize(processed_text)
    
    js_frameworks = re.findall(r'\b\w+\.js\b', text.lower())

    sql_db = re.findall(r"\b((?:\w+)ql(?:\w+)?|(?:\w+)db)\b", text.lower())

    for category, skills in skill_dictionaries.items():
        for skill in skills:
            if ' ' not in skill:
                if skill in ngram_list:
                    clean_skill = skill.replace('.', '')
                    extracted_skills[category].add(clean_skill)
            else:
                skill_words = skill.split()
                for ngram in ngram_list:
                    if skill == ngram:
                        clean_skill = skill.replace('.', '')
                        extracted_skills[category].add(clean_skill)
                        break
                    elif len(ngram.split()) > len(skill_words) and skill in ngram:
                        ngram_words = ngram.split()
                        for i in range(len(ngram_words) - len(skill_words) + 1):
                            if ' '.join(ngram_words[i:i+len(skill_words)]) == skill:
                                clean_skill = skill.replace('.', '')
                                extracted_skills[category].add(clean_skill)
                                break
    
    sliding_matches = sliding_window_match(tokens, skill_dictionaries)
    for category, skills in sliding_matches.items():
        extracted_skills[category].update(skills)
    
    for skill in js_frameworks:
        clean_skill = skill.replace('.', '')
        extracted_skills["Frameworks (FW)"].add(clean_skill)
    
    for skill in sql_db:
        extracted_skills["Databases (DB)"].add(clean_skill)
    
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

sample_text = """
JOHN SMITH
123 Tech Lane, Silicon Valley, CA 94025
Phone: (555) 123-4567
Email: john.smith@email.com

PROFESSIONAL SUMMARY
Experienced Software Engineer with 5+ years of expertise in full-stack development and machine learning applications. Proven track record of delivering scalable solutions and leading cross-functional teams.

WORK EXPERIENCE

Senior Software Engineer
Google, Mountain View, CA
January 2020 - Present
• Led development of cloud-based machine learning pipeline processing 1M+ daily transactions
• Managed team of 6 engineers, improving sprint velocity by 40%
• Implemented microservices architecture using Python and Kubernetes
• Reduced system latency by 60% through optimization of database queries

Software Developer
Microsoft, Seattle, WA
June 2017 - December 2019
• Developed REST APIs serving 500K+ daily users
• Collaborated with product teams to implement new features in Azure platform
• Mentored 3 junior developers and conducted code reviews
• Created automated testing framework reducing QA time by 30%

EDUCATION

Master of Science in Computer Science
Stanford University, Stanford, CA
2015 - 2017
• GPA: 3.9/4.0
• Thesis: "Deep Learning Applications in Natural Language Processing"

Bachelor of Science in Software Engineering
University of California, Berkeley
2011 - 2015
• Dean's List: All semesters
• Minor in Mathematics

SKILLS

Programming Languages:
• Python, Java, JavaScript, C++
• SQL, MongoDB, PostgreSQL
• React, Node.js, Django

Tools & Technologies:
• Docker, Kubernetes, AWS
• TensorFlow, PyTorch
• Git, Jenkins, JIRA

CERTIFICATIONS
• AWS Certified Solutions Architect
• Google Cloud Professional Developer
• Certified Scrum Master

PROJECTS

Machine Learning News Aggregator
• Built news classification system using BERT
• Achieved 95% accuracy in topic categorization
• Deployed on AWS using containerized architecture

Real-time Analytics Dashboard
• Developed dashboard processing 100K+ events/second
• Implemented using React and WebSocket
• Reduced loading time by 70%

LANGUAGES
• English (Native)
• Spanish (Professional)
• Mandarin (Basic)

AWARDS & ACHIEVEMENTS
• Best Technical Innovation Award, Google (2021)
• 1st Place, Microsoft Hackathon (2018)
• Published paper in ACM Conference on ML (2017)
"""

category_abbreviations = {
    "Programming Language (PL)": "PL",
    "Framework (FW)": "FW",
    "Database (DB)": "DB",
    "Cloud Platform (CP)": "CP",
    "DevOps (DO)": "DO",
    "Network & Security (NS)": "NS",
    "Data Analysis & Science (DAS)": "DAS",
    "Software Engineering (SWE)": "SWE",
    "Project Management (PM)": "PM",
    "Education Certification (EC)": "EC",
    "Soft Skills (SS)": "SS",
}
