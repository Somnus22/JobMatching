# JobMatching

### Folders
# Final-service
- Storing the stuff necessary for the final app. 
- Datasets:
  - short.csv is the subset of job postings we are using for testing the app.
  - extracted_job_skills.csv is the result of extracting job skills (done previously) from the short.csv, and used to store the job postings in the app's database.
- app.py: The actual Docker app
- crf_model.joblib: Our saved CRF model for NER
- Dockerfile: Container instructions for app
- matching.model: Our saved model for word embedding matching
- requirements.txt: Library pre-requisites for app
- resources.py: Function definitions + imports for app

# Bert
- What we used to test our BERT models, containing various JSON/txt files for testing purposes, and notebooks for task 1's BERT, task 2's BERT, testing the performance of task 1's BERT, and finetuning the task 1 BERT.

# Match_test
- The combined match scores of the different LLMs and ours, plus a test.ipynb to output the match scores for comparison with the LLMs

# Rule_based_NER
- Our rule based solution for Task 1, including our main model tested on resume, then using the same model for job postings

# TF_IDF_matching
- Just our TF-IDF model for Task 2

# Word_embedding_matching
- Our word2vec model for Task2

# Docker_compose.yml
- Our docker file for running the app easily

# Front_end.html
- Our app's website.
  
### Running the App
1) Start docker and MAMP
2) In the main folder, type docker-compose -f docker_compose.yml up --build
3) Run the html 
4) Upload Resume
5) Extract Skills
6) Load Job Postings
7) Match to Jobs

- If any issues faced with the app in terms of database, close the app, delete docker container and start from step 2. There are unique constraints in place to not overwrite files saved under each name.
