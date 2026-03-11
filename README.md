# ML-Content-Moderation-Pipeline

The below give context to an end-to-end toxic content classification pipeline: 

Text Classification: given some text, assign it a label (toxic, obscene, threatening). The model learns patterns from labelled examples, then predicts labels for new and unseen text (generalisation).

Structure:
1. Data Collection - using a public dataset
2. Preprocessing - cleaning the text
3. Feature extraction - using TF-IDF
4. Model training
5. Evaluation
6. Deployment 

The following concepts will be covered:

TF-IDF: Term Frequency-Inverse Document Frequency, converts text to numbers by measuring how important a word is in a document releative to all documents - turns words into something a model can process.

Logistic Regression: algorithm used to create a decision boundary/threshold to separate classes.

Precision: e.g. of all items flagged toxic by the model, how many actually were

Recall: of all toxic items, how many were caught by the model?

F1: number that balanced both of the above. 

Confusion Matrix: Table showing where model succeeds and fails. 

Multi-label classification: each text can have multiple labels.

- Ranith Simanmeru Pathiranage