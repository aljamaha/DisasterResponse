from sqlalchemy import create_engine
import pandas as pd
import re
import sys
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

def load_data(database_filepath):
	  '''
	  Inputs: database_filepath
	  Outputs:
		  			X: inputs to ML model
		  			Y: outputs to ML model
		  			category_names
	  '''
	  engine    = create_engine('sqlite:///'+database_filepath)   #create connection
	  con       = engine.connect()                                #connect to db    
	  df        = pd.read_sql_table('Figure8', con)               #read data
	  X         = df["message"]
	  Y         = df.drop(labels=["message", "original", "genre"], axis=1)
	  category_names = Y.columns

	  return X, Y, category_names

def tokenize(text):
          '''
	  Inputs: text
	  Outputs: processed and cleaned text
          '''
          text   = text.lower()                                                           #lowercase
          text   = re.sub(r"[^a-zA-Z0-9]", " ", text)                           #remove punctuations
          tokens = word_tokenize(text)                                          #create tokens
          clean = [WordNetLemmatizer().lemmatize(w) for w in tokens] 	# Lemmatise words

          return clean

def build_model():
          '''Outputs: pipeline for ML'''
          basic_logit = LogisticRegression() #LR

          pipeline = Pipeline([
                                        ('vect', CountVectorizer(tokenizer=tokenize)),
                                        ('tfidf', TfidfTransformer()),
                                        ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))
                                     ])
          parameters = {
                'tfidf__smooth_idf':[True, False],
                'clf__estimator__estimator__C': [1, 2, 5]
             }
          cv = GridSearchCV(pipeline,param_grid= parameters,  scoring='precision_samples', cv = 5)

          return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
          '''
	  Inputs: 
 			model: ML model
			X_test: ML datasets input test
 			Y_test: ML datasets output test
          '''

          Y_pred = model.predict(X_test)
    
          for i, col in enumerate(category_names):
                  'model evaluation'    
                  y = list(Y_test.values[:, i])
                  y_pred = list(Y_pred[:, i])
                  target_names = ['is_{}'.format(col), 'is_not_{}'.format(col)] 
                  print(classification_report(y, y_pred, target_names=target_names))

          return 'None'

def save_model(model, model_filepath):
	  '''
	  saves ML model
	  '''
	  joblib.dump(model, model_filepath)

	  return 'None'

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
