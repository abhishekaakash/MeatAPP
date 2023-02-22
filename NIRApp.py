import pickle
import pandas as pd
from flask import Flask, request, render_template
from sklearn import metrics

meatApp = Flask(__name__)


@meatApp.route('/')


@meatApp.route('/index.html')
def home():
    return render_template('index.html')


@meatApp.route('/read_me.html')
def read_me():
    return render_template('read_me.html')



@meatApp.route('/meatType_prediction.html')
def f():
    return render_template('meatType_prediction.html')


@meatApp.route('/cv.html')
def cv():
    return render_template('cv.html')


@meatApp.route('/MLpredict', methods=['POST'])
def MLmeatPredict():

    myModel = 'static/myModel/LOGISTIC.pkl'
    exceptionVar = ''
    myrowSelected = ''


    meat_dfff = pd.read_csv('static/data/meatTest.csv', index_col=0)
    myrowSelected = int(request.form['rowSelected'])


    if ((myrowSelected <= 0) or (myrowSelected > meat_dfff.shape[0])):
        print('Value entered is out of range!!!!!!')
        myrowSelected = int(1)
        exceptionVar = 'Oops! You selected out of range value, getting prediction for default first record.'
    else :
       print('Value is correct!!!!!!')


    meat_test_df2 = pd.DataFrame(meat_dfff.iloc[myrowSelected-1:myrowSelected])
    meat_test_df = meat_test_df2.iloc[:, :3]
    actualLabel = meat_test_df2.iloc[:, -1]
    # print('actualLabel is ',actualLabel[0])

    mlmodel = pickle.load(open(myModel, 'rb'))
    prediction = mlmodel.predict(meat_test_df)


    MLaccuracy = metrics.accuracy_score(actualLabel, prediction)
    predictedLabel = prediction[0]

    # print(prediction)
    return render_template('meatType_prediction.html',meataccuracy=MLaccuracy,exceptionVar = exceptionVar, meatdata=[meat_test_df2.to_html()], columns=[''],
                           Mlresult=f' Prediction meat Target: {predictedLabel}')



@meatApp.route('/MeatDF', methods=("POST", "GET"))
def DFMeat_tab():
    MeatDf = pd.read_csv('static/data/meatTest.csv', index_col=0)
    return render_template('meatType_prediction.html', columns=[''], meatdata=[MeatDf.to_html()])


if __name__ == "__main__":
    meatApp.run(host='0.0.0.0', port=80, debug=True)
