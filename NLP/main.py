# import libraries
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import os
import pandas as pd
import nltk
from nltk import *
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn import model_selection, preprocessing, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

nltk.download('stopwords')

# function to read data from folders
def read_data(folders):
    data = []
    for folder in folders:
        folder_path = os.path.join(r'D:\Lastyear\NLP\txt_sentoken', folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r') as f:
                text = f.read()
            label = folder
            data.append((text, label))
    df = pd.DataFrame(data, columns=['text', 'label'])
    return df

# call the function with the desired arguments
folders = ['neg', 'pos']
df = read_data(folders)

# function to preprocess data
def preprocess_data(df, test_size):
    df['text'] = df['text'].apply(lambda x: x.lower())
    df['text'] = df['text'].str.replace('[^\w\s]', '', regex=True)
    stop = stopwords.words('english')
    df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    st = SnowballStemmer('english')
    df['text'] = df['text'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
    lemmatizer = WordNetLemmatizer()
    df['text'] = df['text'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))
    df = df.sample(frac=1, random_state=42)
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df['text'], df['label'], test_size=test_size, random_state=42)
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.transform(valid_y)
    return train_x, valid_x, train_y, valid_y


# call the function with the desired arguments
test_size = 0.2
train_x, valid_x, train_y, valid_y = preprocess_data(df, test_size)

# create a TF-IDF vectorizer object
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)

# fit the vectorizer on the dataframe text column
tfidf_vect.fit(df['text'])

# function to extract features
def extract_features(df, train_x, valid_x):
    xtrain_tfidf = tfidf_vect.transform(train_x)
    xvalid_tfidf = tfidf_vect.transform(valid_x)
    return xtrain_tfidf, xvalid_tfidf

# call the function with the desired arguments
xtrain_tfidf, xvalid_tfidf = extract_features(df, train_x, valid_x)

# function to train models
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    classifier.fit(feature_vector_train, label)
    predictions = classifier.predict(feature_vector_valid)
    return metrics.accuracy_score(predictions, valid_y), classifier

# Naive Bayes training
accuracy1, naiveclf = train_model(naive_bayes.MultinomialNB(alpha=0.6), xtrain_tfidf, train_y, xvalid_tfidf)
print("Accuracy of Naive Bayes classifier is: ", accuracy1)

# SVM training
accuracy2, svmclf = train_model(svm.SVC(kernel='linear'), xtrain_tfidf, train_y, xvalid_tfidf)
print("Accuracy of SVM classifier is: ", accuracy2)

# Random Forest training
accuracy3, randomclf = train_model(RandomForestClassifier(n_estimators=100), xtrain_tfidf, train_y, xvalid_tfidf)
print("Accuracy of Random Forest classifier is: ", accuracy3)

# Gradient Boosting training
accuracy4, gradclf = train_model(GradientBoostingClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
print("Accuracy of Gradient Boosting classifier is: ", accuracy4)

# Decision Tree training
accuracy5, desclf = train_model(DecisionTreeClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
print("Accuracy of Decision Tree classifier is: ", accuracy5)

# function to print the best model
def print_best_model(models, accuracies):
    best_index = accuracies.index(max(accuracies))
    best_model = models[best_index]
    best_accuracy = accuracies[best_index]
    print(f"The best model is {best_model} with an accuracy of {best_accuracy}")

# list of models and accuracies
models = ['Naive Bayes', 'SVM', 'Random Forest', 'Gradient Boosting', 'Decision Tree']
accuracies = [accuracy1, accuracy2, accuracy3, accuracy4, accuracy5]
print_best_model(models, accuracies)

# define the classifiers and their names
classifiers = [
    ('Naive Bayes', naiveclf),
    ('SVM', svmclf),
    ('Random Forest', randomclf),
    ('Gradient Boosting', gradclf),
    ('Decision Tree', desclf)
]

# initialize lists to store the accuracies
train_accs = []
test_accs = []

# loop through the classifiers
for name, clf in classifiers:
    # calculate the training accuracy
    train_acc = clf.score(xtrain_tfidf, train_y)
    train_accs.append(train_acc)
    # calculate the testing accuracy
    test_acc = clf.score(xvalid_tfidf, valid_y)
    test_accs.append(test_acc)

# create a bar chart to visualize the accuracies
x_pos = [i for i, _ in enumerate(classifiers)]
plt.bar(x_pos, train_accs, color='blue', alpha=0.5, label='Training Accuracy')
plt.bar(x_pos, test_accs, color='red', alpha=0.5, label='Testing Accuracy')
plt.xticks(x_pos, [name for name, _ in classifiers])
plt.legend()
plt.show()

# function to plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# get the predictions for the validation set using the best model (SVM)
y_pred = svmclf.predict(xvalid_tfidf)

# calculate the confusion matrix
cm = confusion_matrix(valid_y, y_pred)

# plot the confusion matrix
plot_confusion_matrix(cm, classes=['neg', 'pos'], normalize=True, title='Confusion Matrix')

# Save the best SVM model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(svmclf, f)

# Save the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vect, f)

print("Models saved successfully!")
