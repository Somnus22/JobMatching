import re
import nltk
from nltk.util import ngrams
from collections import defaultdict

nltk.download('punkt_tab')

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
        extracted_skills["Framework (FW)"].add(clean_skill)
    
    for skill in sql_db:
        extracted_skills["Database (DB)"].add(clean_skill)
    
    return extracted_skills

testResume = """
DAVID CHEN
Software Developer
Boston, MA | david.chen@email.com | (617) 555-1234 | linkedin.com/in/davidchen

SUMMARY
-------
Dedicated Software Developer with 4 years of experience in full-stack web development. Proficient in JavaScript, Python, and Java with expertise in React.js and Django frameworks. Passionate about creating efficient, scalable solutions and continuously learning new technologies.
Database management, Database administration, Database security, Database modelling, Stress testing, load testing, express.js, front-end, decision-making

SKILLS
------
Languages: JavaScript, Python, Java, TypeScript, HTML5, CSS3, SQL, C++, C#. C
Frameworks & Libraries: React.js, Django, Spring Boot, Node.js, Express.js, jQuery, Bootstrap
Tools & Technologies: Git, Docker, AWS (EC2, S3), RESTful APIs, MongoDB, PostgreSQL, Redis, MS SQL SERVER, GRAPHSQL, NOSQL, FIREBASE, gitlab
Methodologies: Agile/Scrum, Test-Driven Development, CI/CD

PROFESSIONAL EXPERIENCE
----------------------
SOFTWARE DEVELOPER | Innovate Solutions | Boston, MA | 2022-Present
- Developed and maintained multiple web applications using React.js and Django, serving 20,000+ users
- Implemented authentication system using OAuth 2.0, improving security and user experience
- Containerized applications using Docker, reducing deployment time by 40%
- Collaborated with UX designers to implement responsive, accessible user interfaces
- Participated in code reviews and mentored junior developers

JUNIOR SOFTWARE DEVELOPER | TechStart Inc. | Cambridge, MA | 2020-2022
- Built RESTful APIs using Django and integrated with front-end React components
- Optimized database queries in PostgreSQL, improving application performance by 30%
- Created automated testing suites with Pytest and Jest, achieving 85% code coverage
- Assisted in migrating legacy PHP application to modern React/Django stack
- Participated in daily stand-ups and bi-weekly sprint planning meetings

EDUCATION
---------
BACHELOR OF SCIENCE IN COMPUTER SCIENCE | Boston University | 2020
- GPA: 3.7/4.0
- Relevant coursework: Data Structures, Algorithms, Database Systems, Web Development
- Senior Project: Developed a task management application using MERN stack

PROJECTS
--------
INVENTORY MANAGEMENT SYSTEM | github.com/davidchen/inventory-app
- Built a full-stack inventory management application using React, Node.js, and MongoDB
- Implemented barcode scanning functionality using WebRTC
- Deployed application to AWS with CI/CD pipeline using GitHub Actions

WEATHER VISUALIZATION DASHBOARD | github.com/davidchen/weather-viz
- Created an interactive dashboard using D3.js to visualize historical weather data
- Integrated with OpenWeatherMap API to fetch and display real-time weather information
- Implemented responsive design principles for mobile compatibility

CERTIFICATIONS
-------------
- AWS Certified Developer – Associate
- MongoDB Certified Developer
- Certified Scrum Master (CSM)

ADDITIONAL INFORMATION
---------------------
- Languages: English (Native), Mandarin Chinese (Fluent)
- Volunteer: Code instructor at local community center teaching Python to high school students
- Interests: Hiking, photography, contributing to open-source projects" \" \
"""