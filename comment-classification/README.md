# Toxic-Content-Filtering
NLP  based project for toxic content filtering

Dataset: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data 

Libraries required:
    pandas
    nltk
    nltk.download(‘averaged-perceptron-tagger’)
    Flask
    numpy
    scikit-learn
    matplotlib

Steps to run:
Before executing any of the following commands, ‘train.csv’ and ‘test.csv’ files of the  dataset must be present in the ‘data’ directory (download them from the link above).
   1. Create and save model
   
        $cd src
        
        $python main.py create
        
        It will create and save pickle files of the model in ‘model’ directory
   2. Generate output file for entire kaggle test set
   
        $cd src
        
        $python main.py test
   3. Test on a single  comment using the developed application
   
        $cd src
        
        $python main.py
        
      The application can be accessed from http://localhost:5000/ 

