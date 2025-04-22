# COVID-19 Detection Project: Presentation Script

## Introduction to the Project (Speaker 1)

### Slide 1: Title Slide
Good afternoon everyone! Today we're excited to present our COVID-19 Detection Project. This project represents a comprehensive approach to detecting COVID-19 from unstructured medical text using artificial intelligence techniques. I'm [Speaker 1], and I'll be presenting along with my colleagues [Speaker 2] and [Speaker 3].

### Slide 2: Problem Statement
During the COVID-19 pandemic, healthcare professionals faced a critical challenge: how to quickly identify potential COVID cases when symptoms overlap with other respiratory conditions like the flu or common cold. 

With limited testing resources and delays in test results, clinicians needed better tools to help them make informed decisions based on the information they already had - primarily patient descriptions and clinical notes.

The key question we aimed to answer was: Can we leverage NLP and machine learning to analyze unstructured medical text and accurately detect COVID-19 cases?

### Slide 3: Project Significance
The significance of this work extends beyond COVID-19. Many diseases present with overlapping symptoms, and clinicians routinely make initial assessments based on textual information before test results are available.

By successfully applying NLP to COVID-19 detection, we're demonstrating a technique that could be adapted to other medical conditions, potentially transforming how we use the vast amounts of unstructured medical text that exist in healthcare systems worldwide.

I'll now pass it over to [Speaker 2] who will explain our technical approach.

## Technical Approach (Speaker 2)

### Slide 4: Technical Pipeline Overview
Thank you, [Speaker 1]. Our approach to this problem leveraged a two-stage pipeline that combines Named Entity Recognition with transformer-based classification.

As you can see on this slide, the pipeline starts with raw clinical notes, extracts medical entities using NER, converts these entities to structured features, integrates them with other patient data, and finally uses a BioBERT model to predict COVID-19 likelihood.

This combination allows us to capture both the presence of symptoms and their contextual information - such as severity and temporal characteristics.

### Slide 5: Data Sources
Our project utilized multiple data sources:

First, we used the MIMIC-IV clinical dataset containing de-identified patient records and clinical notes. This provided realistic medical text with the complexity we'd expect in real-world applications.

Second, we incorporated CDC COVID-19 case surveillance data to understand typical symptom patterns and prevalence.

Finally, we used the CORD-19 research corpus, which provided valuable context from scientific literature about COVID-19 symptoms and progression.

### Slide 6: Named Entity Recognition (NER)
The first key component of our pipeline is Named Entity Recognition. We implemented three different NER approaches, each with its own strengths:

1. A rule-based system using regular expressions and keyword matching - simple but fast
2. A spaCy-based NER model that we could custom train
3. A transformer-based NER using biomedical pre-trained models

These systems extract three main entity types from clinical notes:
- Symptoms like fever, cough, or loss of taste
- Time expressions that indicate when symptoms began or how long they've lasted
- Severity indicators that describe symptom intensity

Let me show you a quick example of what our NER system extracts from a typical clinical note...

### Slide 7: NER Example
Here's an example from our system. Given this clinical note about a 45-year-old patient, our NER system identifies key entities:
- Symptoms: fever, dry cough, fatigue, loss of taste, loss of smell
- Time expressions: "for the past 3 days", "since yesterday"
- Severity: none explicitly mentioned in this example

These extracted entities then become features for our classification model, along with structural information about how these symptoms relate to each other temporally.

I'll now hand it over to [Speaker 3] who will explain our classification approach and results.

## Model Architecture and Results (Speaker 3)

### Slide 8: BioBERT Classification
Thank you, [Speaker 2]. For the classification stage of our pipeline, we fine-tuned a BioBERT model specifically for COVID-19 detection.

BioBERT is a version of BERT that's been pre-trained on biomedical literature, making it particularly suitable for understanding medical terminology and contexts.

Our implementation used the dmis-lab/biobert-base-cased-v1.1 model as a foundation, which has 12 transformer layers, 768 hidden dimensions, and 12 attention heads.

We fine-tuned this model on our integrated dataset of NER features and structured patient data, optimizing for COVID-19 classification performance.

### Slide 9: Performance Metrics
Our BioBERT model achieved impressive performance metrics:
- 92.8% accuracy on our test set
- Precision of 0.94, meaning when it predicts COVID-19, it's right 94% of the time
- Recall of 0.91, meaning it correctly identifies 91% of actual COVID-19 cases
- F1 score of 0.925, which balances precision and recall
- ROC AUC of 0.967, showing excellent discrimination ability

These metrics significantly outperform traditional machine learning approaches, as you can see in the comparison chart. While gradient boosting came close with an ROC AUC of 0.937, BioBERT still provided the best overall performance.

### Slide 10: Error Analysis
We performed a detailed error analysis to understand when our model fails and why.

Most false negatives occurred in cases with atypical symptom presentation - patients who had COVID-19 but presented with unusual symptoms or very mild cases.

Most false positives happened with other respiratory conditions that closely mimic COVID-19, particularly influenza cases with fever and respiratory symptoms.

This analysis helps us understand the model's limitations and guides future improvements.

Now I'll hand back to [Speaker 1] to introduce the practical application of our work: the Harvey chatbot.

## Harvey Chatbot (Speaker 1)

### Slide 11: Harvey Chatbot Introduction
Thank you, [Speaker 3]. While the technical components of our project are impressive, we wanted to create something practical that healthcare professionals could actually use. That's why we developed "Harvey" - named after William Harvey, the physician who discovered blood circulation.

Harvey is a chatbot interface that implements our COVID-19 detection pipeline in an interactive format. It allows medical professionals to explore patient data through natural conversation, highlighting relevant symptoms and providing COVID-19 risk assessments.

### Slide 12: Harvey Interface Screenshot
Here you can see the Harvey interface, which features a clean, minimalist design inspired by modern AI assistants. On the left is a patient selection panel showing demographics and risk levels. On the right is the chat interface where users can ask questions about the patient.

[This slide includes screenshots of the Harvey chatbot interface]

### Slide 13: Key Features
Harvey has several key features designed for the clinical environment:

- Interactive patient records with COVID probability scores
- Natural language querying of patient symptoms and lab results
- Automated extraction of key entities from clinical notes
- Visualized risk assessment based on our BioBERT model
- Suggestion bullets for common clinical questions
- Fallback responses to ensure reliability even when the backend has issues

Let's look at how Harvey processes a typical interaction...

### Slide 14: Interaction Example
In this example interaction, a clinician asks about a patient's key symptoms. Harvey processes this query and:
1. Extracts the intent (looking for symptoms)
2. Identifies relevant entities in the patient's clinical notes
3. Formulates a response highlighting the most important symptoms
4. Provides additional context about symptom onset and progression
5. Calculates a COVID-19 probability based on the symptom pattern

This natural interaction makes the underlying AI accessible and practical for busy healthcare environments.

I'll now hand back to [Speaker 2] to discuss implementation challenges and future directions.

## Technical Challenges and Future Work (Speaker 2)

### Slide 15: Implementation Challenges
Thank you, [Speaker 1]. Implementing this project wasn't without challenges. Here are some of the key difficulties we faced:

1. Working with de-identified clinical text that sometimes lacked standardization
2. Handling class imbalance in our training data
3. Developing robust NER that could handle the variety of ways symptoms are described
4. Ensuring privacy compliance when working with medical data
5. Creating a reliable web interface with appropriate fallbacks

We addressed these through techniques like data augmentation, ensemble NER approaches, and multi-level fallback systems in the chatbot.

### Slide 16: Technical Architecture
From an implementation perspective, our system uses a modular architecture:

- The NER components are implemented in Python with multiple extraction strategies
- BioBERT is integrated using the Hugging Face transformers library
- The Harvey chatbot uses a Flask backend with JavaScript/HTML/CSS frontend
- The system uses a RESTful API pattern for communication between components

This modular design makes it easy to update or replace individual components as technology evolves.

### Slide 17: Future Directions
Looking ahead, we see several promising directions for this work:

1. Multimodal integration - combining text analysis with medical imaging data
2. Variant identification - retraining models to detect specific COVID variants
3. Severity prediction - not just detecting COVID but predicting disease progression
4. Temporal modeling - tracking symptom changes over time
5. Expanded clinical deployment - implementing Harvey in more healthcare settings

We believe the techniques developed here have broad applicability beyond COVID-19.

I'll now hand back to [Speaker 3] for our conclusion.

## Conclusion (Speaker 3)

### Slide 18: Project Summary
Thank you, [Speaker 2]. To summarize what we've accomplished in this project:

We've developed a complete pipeline for COVID-19 detection from unstructured medical text, combining NER with BioBERT classification.

We've demonstrated strong performance metrics, with our model achieving 92.8% accuracy and outperforming traditional machine learning approaches.

We've created Harvey, a practical chatbot interface that makes this technology accessible to healthcare professionals.

### Slide 19: Key Takeaways
The key takeaways from our project are:

1. NLP techniques can effectively extract valuable diagnostic information from unstructured medical text
2. Combining NER with transformer models creates a powerful pipeline for medical text analysis
3. Domain-specific transformers like BioBERT offer significant advantages for medical applications
4. Interactive interfaces like Harvey make AI more accessible in clinical settings
5. This approach can potentially be adapted to other medical conditions beyond COVID-19

### Slide 20: Thank You & Questions
Thank you all for your attention today. We're excited about the potential impact of this work and grateful for the opportunity to share it with you.

We're now happy to take any questions you might have about our approach, implementation, or results.

[End of Presentation]