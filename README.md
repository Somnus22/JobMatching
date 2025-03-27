# FloppyDisk

### What is our project about

Hiring managers have a hard time efficiently and effectively screening large volumes of resumes against job requirements. With potentially hundreds of resumes received for each position, manually reviewing and matching prospective candidates to job requirements is time-consuming and potentially inconsistent. The unstructured nature of both resumes and job postings makes it difficult to quickly identify good candidates, leading to increased time-to-hire and potential missed opportunities.

Hence, this project aims to develop a text analysis solution that automatically generates accurate job matching scores between resumes and job postings, helping hiring managers quickly identify the most promising candidates.

### Exploratory Data Analysis (EDA)

| Dataset | Description |
| ------- | ----------- |
| [Resume Corpus](https://github.com/florex/resume_corpus) | Contains a collection of resumes |
| [LinkedIn Job Postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) | Contains a collection of job postings |

We carried out simple data and text preprocessing prior to obtaining the statistical data required for the EDA. 

Data Preprocessing (Cleaning and Standardization)
- Removed HTML tags, encoding artifacts, special characters, company disclaimers.

Text Preprocessing (For NLP Tasks)
- Named Entity Recognition (NER) Analysis: Extracted skills, job titles, companies.
- Word Cloud Visualization: Most frequent words and extracted skills.
- POS Tagging Distribution: Analysis of nouns, verbs, adjectives in text.

## Text Mining Tasks

### Task 1: Named Entity Recognition (NER)
**Model 1: Rule-Based NER with spaCy, NLTK, Regex**
- **Techniques:** Patterns, dictionaries, regex
- **Steps:** Create dictionaries → Define patterns → Apply rules → Extract entities

**Model 2: Statistical NER with sklearn-crfsuite**
- **Techniques:** Feature engineering, sequence labeling
- **Steps:** Label data → Extract features → Train CRF → Apply model

**Model 3: Deep Learning NER (BERT) with Hugging Face transformers, Pytorch**
- **Techniques:** Neural networks, transfer learning
- **Steps:** Load BERT → Fine-tune → Create pipeline → Extract entities

**Evaluation Metrics**
- **Precision:** Measures how many extracted entities are correct. E.g., Identifying "Python" as a programming skill and "AWS" as a cloud technology.
- **Recall:** Measures how many relevant skills were extracted. E.g., Finding all required and preferred skills mentioned in a job posting like "Python", "AWS", "Docker".
- **F1-score:** Combines precision and recall. E.g., Model correctly identifies skills and their requirement levels (required, preferred, good to have).
- **Entity classification quality (Qualitative):** Correct identification of skills.


### Task 2: Document Similarity
**Model 1: TF-IDF Similarity with sklearn**
- **Techniques:** Term frequency, cosine similarity
- **Steps:** Convert to vectors → Calculate similarity → Generate scores

**Model 2: Word Embeddings with gensim, Word2Vec**
- **Techniques:** Word embeddings, semantic vectors
- **Steps:** Generate embeddings → Create vectors → Calculate similarity → Generate scores

**Model 3: BERT Embeddings with sentence-transformers**
- **Techniques:** Contextual embeddings, neural similarity
- **Steps:** Generate embeddings → Calculate similarity → Weight features → Generate scores

**Evaluation Metrics**
- **Precision:** Measures how many matched resumes are truly similar
- **Recall:** Measures how many relevant resumes are retrieved compared to all possible relevant resumes
- **F1-score:** Harmonic mean of precision and recall, balancing false positives and false negatives
- **Mean Reciprocal Rank (MRR):** Evaluates ranked retrieval performance, checking how high relevant resumes are ranked
