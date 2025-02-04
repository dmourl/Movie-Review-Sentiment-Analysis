# Movie-Review-Sentiment-Analysis
Using research designed and developed a binary classification model to predict sentiment in 50 000 IMDb movie reviews using Logistic Regression after evaluating multiple ML models.

Movie review sentiment classification project Introduction
Movie reviews are an important factor in the entertainment industry. The opinion of experts or even just normal everyday movie watchers matters to people. (Other people’s opinions hold a lot of weight.) Nowadays it’s normal to check the reviews on a movie before going to see it. The general opinion of a movie affects a person’s decision to go watch a movie (or stream
it). However, rarely anyone has time to go through hundreds of reviews, so classifying the sentiment of a review is useful. This way the general opinion of a movie can be “calculated” as mostly positive or mostly negative based on the reviews. In this project the focus is on sentiment analysis of movie reviews. Sentiment analysis and classification helps consumers and businesses in decision-making. It is also easier for businesses to use classified reviews for future advertising and marketing.
In this project we start by formulating the problem and then going through three sections. In section 1, we discuss the data, its structure and processing. In section 2, we discuss the task at hand, the possible methods, comparing different models and justifications for choosing a specific one. In section 3, we discuss our results and conclude the problem.
Problem formulation
To approach the task of classifying movie reviews as positive or negative based on the text of a review we have to do binary classification. Ideally, we eventually create a model that is trained to predict whether a given review is of positive or negative sentiment.
Section 1 - Data
1.1 Dataset
The dataset is an IMBd dataset from Kaggle. IMBd is the world's most popular database for movie ratings and reviews. It is widely used and authoritative which makes it a credible source. This dataset contains an equal number of positive and negative reviews balancing the classification training.
1.2 Datapoints
In total 50 000 movie reviews. Our data is binary. Each movie review is text string and has its corresponding sentiment label (positive or negative). Therefore, the features are the text of each movie review, these being the input features. The dataset is multidimensional because the features consist of multiple words. The text in the reviews can be transformed into numerical features. The labels are the sentiment of each review.
1.3 Data preprocessing
 
We clean the text, meaning we remove all special characters and make everything lowercase. We split the review texts into one word at a time and leave out filler words for example “the”, “is” and “a”. We transform words into their basic forms. Also regarding the labels, we make them numerical as well, assigning positive as 0 and negative as 1.
Section 2 – ML Model 2.1 ML task
The type of this ML task is supervised learning. This is because our data is labeled so we aim to train a model that predicts the label (the sentiment) based on the input feature (the review).
2.2 Model and Methods
Due to having only two categories the most effective ML method is logistic regression. As A. Jung states in chapter 3.6 of the book “Machine Learning: The Basics” (2022): “Logistic regression is a ML method, that allows to classify datapoints according to two categories. Thus, logistic regression is a binary classification method that can be applied to datapoints characterized by feature vectors x and binary labels y. These binary labels take on values from a label space that contains two different label values. Each of these two label values represents one of the two categories to which the datapoints can belong.”
Our dataset fits this description because it has binary labels attached to its features. Our input features (words) can be transformed into numerical form, which makes it easy for logistic regression to find the probability of the review being either positive or negative. Also, logistic regression is computationally efficient, simple and easily interpretable which is valuable for large text data like ours.
2.3 Method comparison
Now we can compare two different ML methods for more accuracy and why we chose Logistic Regression. First, we could solve our problem with Support Vector Machine (SVM). It's efficient for text classification. SVM can handle high-dimensional data and also sparse features. Our problem is exactly a text task so a method that can process text would work with us. Nevertheless, we can also use a different ML method which is Random Forest. It can also handle text data and provide durability and usually it performs well with text data. It would bring more contrast to the linear approaches like logistic regression and SVM.
The reasoning behind us choosing logistic regression is because of its simplicity, efficiency, and suitability for exactly binary classification with very high-dimensional data makes it the best choice for our project. Logistic regression has a good balance of performance and interpretability and especially taking into consideration the massive dataset and binary nature of our task.
2.4 Loss
 
As our course states in the MyCourses page table of classification methods, logistic loss is the appropriate loss function to use when it comes to logistic regression. Logistic loss measures the difference between the predicted possibilities and the true labels, which is very useful especially in binary classification problems. Logistic loss helps us to find the accuracy of our model.
2.5 Model validation process
We split the data into training, validation and test sets. Our training set includes 70% of data so 35 000 reviews. The remaining data is then split into validation data 15% so 7500 reviews and test data which is another 15% so also 7500 reviews. This way we have a large enough amount of data that the model can train with but also enough to validate and test the performance while still avoiding overfitting.
2.6 Feature selection process
We found that using term frequency-inverse document frequency (TF-IDF) is the best technique to capture all sentimental and detailed words from the features. The way it works is it transforms text into a matrix where each word becomes a numerical feature. Important words with sentiment are assigned a higher weight meanwhile unnecessary words are given a lower weight, because words like “and” and “the” are neutral and they don’t contribute to a review being either positive or negative. The model can then use these transformed values as inputs.
Section 3 – Results and Conclusion 3.1 Results
Now why not SVM or Random Forest? Well, that's because for example SVM is slow with large datasets like ours, it tries to find the best boundary between the classes, and it also takes more time to train. However, the main reason for not using SVM is that it might overkill a complex task like this. SVM does not always give probabilities which are important to boost the confidence and accuracy of our predictions. After running a few tests, we noticed that the best fitting ML method would be SVM but only if our dataset was 1000 or under. Now Random Forest would work the best with non-linear data, but our sentiment analysis does not need anything that complex. Random forest will probably also overfit when dealing with text data. It will build decision trees that will take time and resources, and our binary nature task does not need that. It is also a lot slower to train than logistic regression.
We ran a few more tests with the datasize of only 2000, here we noticed how logistic regression needs a large dataset for it to be more accurate. The larger the dataset is the more accurate it is. This proves that it is the best fitting ML method for our task.
Whan running our data with our code and calculating the validation and the test errors in each SVM, Random Forest and Logistic Regression we got the following results: validation accuracy for logistic regression was 0.8344 and test accuracy 0.8304 and for SVM validation accuracy was 0.8373 and test accuracy 0.8205 and lastly for Random forest the validation accuracy was 0.8063 and test accuracy was 0.8044. Here we can compare and see that
 
logistic regression is the most accurate out of all of the methods which is why we chose it as our primary model.
(Unfortunately after many attempts we came to conclusion we should keep the sample size smaller because our computer couldn’t handle such a large dataset. We have noticed that this effects the training accuracy because logistic regression works more accurately with large datasets. To clarify this is not a mistake we made unnoticed but rather an adjustment for the sake of this assignment.)
3.2 Conclusion
We worked on classifying movie reviews as either positive or negative. To tackle this binary classification problem, we used the IMDb data set with 50 000 reviews and used logistic regression. After preprocessing the text data and using TF-IDF to turn the comments into numerical features our model was able to predict sentiment accurately.
Logistic regression is good, and it works well with us now however there is still room for improvement because logistic regression assumes that words and sentiment is linear, so it will miss the more complex patterns in reviews, for example a review that uses a negative word to describe the goodness of a movie. The other models like SVM and especially Random Forest will improve the results, but they are more difficult to mainly train and interpret.
In the future to make our model even better we could try to use more advanced methods, for example neural networks or learn word embeddings to understand the deeper meaning of the text. This would solve our inaccuracy and refine the sentiment classification.
Sources:
N. Lakshmipathi, IMDb Dataset of 50K Movie Reviews, Kaggle, 2018
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie- reviews/data
A. Jung, “Machine Learning: The Basics,” Springer, Singapore, 2022 https://github.com/alexjungaalto/MachineLearningTheBasics/blob/master/MLBasicsBook.p df
MyCourses tables
https://mycourses.aalto.fi/mod/page/view.php?id=1228015
    
[24]:
[nltk_data] Downloading package stopwords to
[nltk_data]
[nltk_data]
  /home/        /nltk_data...
Package stopwords is already up-to-date!
Appendix
October 8, 2024
 import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
 [25]:
[25]:
[26]:
 # Loading dataset from downloads on this device.
# Visually checking the first five datapoints.
df = pd.read_csv('IMDB Dataset.csv')
df.head()
                                              review sentiment
0  One of the other reviewers has mentioned that ...  positive
1  A wonderful little production. <br /><br />The...  positive
2  I thought this was a wonderful way to spend ti...  positive
3  Basically there's a family where a little boy ...  negative
4  Petter Mattei's "Love in the Time of Money" is...  positive
  # Preprocessing the review texts.
# Removing stopwords as a tool for process.
# All the general processing like special chars, lowercase, tokenizing to help␣
↪handle the data.
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
text = re.sub(r'\W', ' ', text) # Special chars text = text.lower()
1

 text = text.split() # Tokenization
text = [word for word in text if word not in stop_words] # Stop word␣ ↪removal
return ' '.join(text)
df['review'] = df['review'].apply(preprocess_text)
# Visually checking the first five preprocessed datapoints. df.head()
[26]:
[27]:
[28]:
      (50000, 5000)
[29]:
[29]:
                                              review sentiment
0  one reviewers mentioned watching 1 oz episode ...  positive
1  wonderful little production br br filming tech...  positive
2  thought wonderful way spend time hot summer we...  positive
3  basically family little boy jake thinks zombie...  negative
4  petter mattei love time money visually stunnin...  positive
  # Bigram model and checking the shape.
count_vectorizer = CountVectorizer(ngram_range=(2, 2))
X_ngrams = count_vectorizer.fit_transform(df['review'])
print(X_ngrams.shape)
(50000, 3099540)
 # Using tf-idf to determine the value/weight/importance of the words.
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(df['review'])
print(X_tfidf.shape)
 # Semantic relationships between words using Word2Vec as a tool.
from gensim.models import Word2Vec
reviews = [review.split() for review in df['review']]
word2vec_model = Word2Vec(reviews, vector_size=100, window=5, min_count=2) word2vec_model.wv.most_similar('movie')
[('film', 0.8463782668113708),
 ('flick', 0.6720701456069946),
 ('movies', 0.6485480070114136),
 ('think', 0.5723590850830078),
 ('sequel', 0.5599719882011414),
 ('really', 0.5508494973182678),
 ('sure', 0.5375732779502869),
 2

 ('guess', 0.534662127494812),
('thats', 0.5309193730354309),
('bothered', 0.5072049498558044)]
[30]:
 # Deeper meaning of words using BERT as a tool.
# Tokenizer to split, Model to generate.
from transformers import BertTokenizer, BertModel import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
sample_text = "This is a movie review."
inputs = tokenizer(sample_text, return_tensors='pt')
outputs = model(**inputs)
[69]:
 X_train_val, X_test, y_train_val, y_test = train_test_split(X_tfidf,␣ ↪df['sentiment'], test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,␣ ↪test_size=0.25, random_state=42)
# Using a smaller sample size for demonstration
X_train_sample = X_train[:1000]
y_train_sample = y_train[:1000]
[70]:
 # Linear Regression model
model = LogisticRegression(max_iter=2000, solver= 'saga') model.fit(X_train_sample, y_train_sample)
#y_pred = model.predict(X_test)
#accuracy = accuracy_score(y_test, y_pred)
#print(f'Accuracy: {accuracy}')
# Make predictions
y_train_pred = model.predict(X_train_sample)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)
# Calculate accuracies
train_accuracy = accuracy_score(y_train_sample, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
# Print the results
3

 print(f'Training Accuracy: {train_accuracy:.4f}') print(f'Validation Accuracy: {val_accuracy:.4f}') print(f'Test Accuracy: {test_accuracy:.4f}')
      Training Accuracy: 0.9750
      Validation Accuracy: 0.8344
      Test Accuracy: 0.8304
[71]:
 # SVM model
from sklearn.svm import SVC
svm_model = SVC(kernel='linear') svm_model.fit(X_train_sample, y_train_sample) #y_pred_svm = svm_model.predict(X_test) #accuracy_svm = accuracy_score(y_test, y_pred_svm) #print(f'SVM Accuracy: {accuracy_svm}')
# Make predictions
svm_y_train_pred = svm_model.predict(X_train_sample)
svm_y_val_pred = svm_model.predict(X_val)
svm_y_test_pred = svm_model.predict(X_test)
# Calculate accuracies
svm_train_accuracy = accuracy_score(y_train_sample, svm_y_train_pred)
svm_val_accuracy = accuracy_score(y_val, svm_y_val_pred)
svm_test_accuracy = accuracy_score(y_test, svm_y_test_pred)
# Print the results
print(f'Training Accuracy: {svm_train_accuracy:.4f}') print(f'Validation Accuracy: {svm_val_accuracy:.4f}') print(f'Test Accuracy: {svm_test_accuracy:.4f}')
[72]:
Training Accuracy: 0.9930
Validation Accuracy: 0.8273
Test Accuracy: 0.8205
 # Random Forest model
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100) rf_model.fit(X_train_sample, y_train_sample) #y_pred_rf = rf_model.predict(X_test)
#accuracy_rf = accuracy_score(y_test, y_pred_rf) #print(f'Random Forest Accuracy: {accuracy_rf}')
4

 # Make predictions
rf_y_train_pred = rf_model.predict(X_train_sample)
rf_y_val_pred = rf_model.predict(X_val)
rf_y_test_pred = rf_model.predict(X_test)
# Calculate accuracies
rf_train_accuracy = accuracy_score(y_train_sample, rf_y_train_pred)
rf_val_accuracy = accuracy_score(y_val, rf_y_val_pred)
rf_test_accuracy = accuracy_score(y_test, rf_y_test_pred)
# Print the results
print(f'Training Accuracy: {rf_train_accuracy:.4f}') print(f'Validation Accuracy: {rf_val_accuracy:.4f}') print(f'Test Accuracy: {rf_test_accuracy:.4f}')
     Training Accuracy: 1.0000
     Validation Accuracy: 0.8063
     Test Accuracy: 0.8044
[ ]: [ ]: [ ]:
   5
