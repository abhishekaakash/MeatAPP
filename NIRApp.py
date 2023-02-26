import pickle
import pandas as pd
import numpy as np
import os
from flask import Flask, request, render_template
from sklearn import metrics
from genetic_selection import GeneticSelectionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from genetic_selection import GeneticSelectionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_decomposition import PLSRegression

data_upload = 'static/upload'
meatApp = Flask(__name__)
meatApp.config['data_upload'] = data_upload

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
    focus_div = True
    if os.path.isfile('static/upload/meats.csv'):
        error_message   = "Exception! , Please hit button 'Process Meat Data'  and then Retry!  "
    else:
        error_message =  "Exception! , Please first upload the data : dataset meats.csv and then re process!  "


    myModel = 'static/upload/PLSnew.pkl'
    exceptionVar = ''
    myrowSelected = ''
    error_code = 0
    error_codeData = 0
    try:
        testDF = pd.read_csv('static/upload/meatTest_PLS.csv', index_col=0)
        myrowSelected = int(request.form['rowSelected'])
        print('Record selected ',myrowSelected)

        if ((myrowSelected < 0) or (myrowSelected > testDF.shape[0])):
            print('Value entered is out of range!!!!!!')
            myrowSelected = int(1)
            exceptionVar = 'Oops! You selected out of range value, getting prediction for default first record.'
        else :

           myrowSelected= myrowSelected +1
           print('Value is correct!!!!!!')

        meat_test_df2 = pd.DataFrame(testDF.iloc[myrowSelected -1:myrowSelected])

        meat_test_df = meat_test_df2.iloc[:, :5]
        actualLabel = meat_test_df2.iloc[:, -1]

        meat_test_df['type'] = actualLabel

        plsmlmodel = pickle.load(open(myModel, 'rb'))
        testDF_noTYPE = testDF.iloc[myrowSelected-1 :myrowSelected].copy()
        testDF_noTYPE = testDF_noTYPE.drop("type", axis='columns')

        print(testDF_noTYPE)
        print('Before prediction, plsmlmodel: ',plsmlmodel)

        predictionPLS = plsmlmodel.predict(testDF_noTYPE)

        print('After prediction')
        keys = ['Beef', 'Chicken', 'Lamb', 'Pork', 'Turkey']

        p = predictionPLS.flatten().tolist()

        data_dict = dict(zip(keys, p))

        predictedLabel = max(data_dict, key=data_dict.get)

        print('actual label', actualLabel.values)

        print('predicted label', predictedLabel)

        if predictedLabel == actualLabel.values:
            MLaccuracy = 1
        else:
            MLaccuracy = 0

        print('MLaccuracy', MLaccuracy)
        pass

    except FileNotFoundError:

        return render_template('meatType_prediction.html',error_message =f' {error_message}',error_codeData = 1 )


    # print(prediction)
    return render_template('meatType_prediction.html',meataccuracy=MLaccuracy,exceptionVar = exceptionVar, meatdata=[meat_test_df.to_html()], columns=[''],
                           Mlresult=f' ML Predicted Meat Type: {predictedLabel}',error_codeData = 0,focus_div=focus_div)



@meatApp.route('/MeatDF', methods=("POST", "GET"))
def DFMeat_tab():
    error_codeData = 0

    if os.path.isfile('static/upload/meats.csv'):
        error_message   = "Exception! , Please hit button 'Process Meat Data'  and then Retry!  "
    else:
        error_message =  "Exception! , Please first upload the data : dataset meats.csv and then re process!  "


    try:
        MeatDf = pd.read_csv('static/upload/meatTest_PLS.csv', index_col=0)

        meat_user_viewDF = MeatDf.iloc[:, :5]
        meat_user_viewDF['type'] = MeatDf.iloc[:, -1]

        pass

    except FileNotFoundError:

        return render_template('meatType_prediction.html',error_message =f' {error_message}',error_codeData = 1 )

    return render_template('meatType_prediction.html', columns=[''], meatdata=[meat_user_viewDF.to_html()])


@meatApp.route('/upload',methods=['GET','POST'])
def upload():
    error_code = 0
    if request.method == 'GET':

        print('request method is GET')

    else:

        error_message = 'Exception! , Please browse the file meats.csv and then Retry!'
        try:

            print('request method is POST')
            meatfile = request.files['file']
            meatfilename = meatfile.filename
            print('meatfilename  :  ',meatfilename)
            print(f" upload folder path is ) {meatApp.config['data_upload']}")
            meatfile.save(os.path.join(meatApp.config['data_upload'],meatfilename))

            pass

        except FileNotFoundError:

            return render_template('meatType_prediction.html', error_message=f' {error_message}', error_code=2)

        return render_template('meatType_prediction.html',result='DataUploaded successfully.')


@meatApp.route('/processML',methods=['GET','POST'])
def processML():
    error_code = 0
    if os.path.isfile('static/upload/meats.csv'):
        error_message   = "Exception! , Please hit button 'Process Meat Data'  and then Retry!  "
    else:
        error_message =  "Exception! , Please first upload the data : dataset meats.csv and then re process!  "

    try:
        meatDF = pd.read_csv("static/upload/meats.csv")
        # Getting meat data without Label
        meat_nolabel_df = meatDF.iloc[:, 1:1050].copy()

        # Getting the column with just label
        meat_label = meatDF.iloc[:, -1].copy()

        meatdf_TRAIN_X, meatdf_TEST_X, meatlabel_trainY, meatlabel_testY = train_test_split(meat_nolabel_df, meat_label,
                                                                                        stratify=meat_label,
                                                                                        test_size=0.2,
                                                                                        random_state=1121218)

        n = len(meatdf_TRAIN_X)

    # 10-fold CV, with shuffle
        kfold_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)

        mse = []

        meatlabel_trainY = meatlabel_trainY.reset_index(drop=True)

        Meatlabel_encoder = LabelEncoder()
        meatlabel_trainY_F = Meatlabel_encoder.fit_transform(meatlabel_trainY)

        algo = PLSRegression(n_components=14)

        meatlabel_trainY = meatlabel_trainY.reset_index(drop=True)
        meatlabel_trainY_F = pd.get_dummies(meatlabel_trainY)

        meatdf_TRAIN_X = meatdf_TRAIN_X.reset_index(drop=True)

        algo.fit(meatdf_TRAIN_X, meatlabel_trainY_F)

    # Downloading the model

        mklFile = open('static/upload/PLSnew.pkl','wb')
        pickle.dump(algo,mklFile)
        mklFile.close()

    ##Downloding the test data for UI
        testDF = meatdf_TEST_X.copy().reset_index(drop=True)
        testDF['type'] = meatlabel_testY.reset_index(drop=True)
        testDF.insert(0, '', range(len(testDF)))
        testDF.to_csv(r'static/upload/meatTest_PLS.csv', index=False)

        Y_pred = algo.predict(meatdf_TEST_X)

        y_test_f = pd.get_dummies(meatlabel_testY.reset_index(drop=True))

        df2 = pd.DataFrame(data=None, columns=range(len(y_test_f.columns)), index=y_test_f.index).fillna(0)
        for i, k in df2.iterrows():
            j = np.argmax(Y_pred[i])
            df2[j][i] = 1

        df2.rename(columns={0: 'Beef', 1: 'Chicken', 2: 'Lamb', 3: 'Pork', 4: 'Turkey'}, inplace=True)

        pls_confusion_t = metrics.confusion_matrix(y_test_f.values.argmax(axis=1), df2.values.argmax(axis=1))
        # print(pls_confusion_t)

        Accuracy = metrics.accuracy_score(y_test_f, df2)


        pass

    except FileNotFoundError:

        return render_template('meatType_prediction.html', error_message =f' {error_message}',error_code = 1 )

    return render_template('meatType_prediction.html', MLAccuracy=f' ML Accuracy is : {Accuracy}')













if __name__ == "__main__":
    meatApp.run(host='0.0.0.0', port=80, debug=True)
