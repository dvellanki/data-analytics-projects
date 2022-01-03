import sys
import model
from flask import Flask, render_template, request

app = Flask(__name__)

def create_model():
    model.create_and_save()

def default_main():
    model.predict_score()

@app.route("/", methods=['GET','POST'])
def welcome():
    return render_template('index.html')

@app.route("/messageprobability", methods=['GET','POST'])
def main():
    message = request.form['message']
    predictions = model.predict_individual_score(message)
    print ("Message: "+message)
    print ("Predictions: ")
    print (predictions)
    return render_template('predictions.html', message=message, predictions=predictions)

if __name__ == '__main__':
    if (len(sys.argv) == 2):
        if(sys.argv[1] == 'create'):
            create_model()
        elif(sys.argv[1] == 'test'):
            default_main()
        else:
            print ('Usage:\tmain.py <Create Model["create"] / Test on full test set["test"]>')
            sys.exit(0)
    else:
        app.run()
