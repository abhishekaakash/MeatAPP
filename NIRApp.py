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
    # Called by Button : Predict , page : meatType_prediction.html
    # This method is going to get the row number passed by user. It will validate if the row number is within the limit
    # or not with respect to the test dataset
    # Method will use the ML Model from Upload folder and then return the prediction label to the calling page meatType_prediction

    focus_div = True
    if os.path.isfile('static/upload/meats.csv'):
        error_message = "Exception! , Please hit button 'Process Meat Data'  and then Retry!  "
    else:
        error_message = "Exception! , Please first upload the data : dataset meats.csv and then re process!  "

    myModel = 'static/upload/PLSnew.pkl'
    exceptionVar = ''
    myrowSelected = ''
    error_code = 0
    error_codeData = 0

    try:

        # Reading the testdata set
        testDF = pd.read_csv('static/upload/meatTest_PLS.csv', index_col=0)
        myrowSelected = int(request.form['rowSelected'])
        print('Record selected ', myrowSelected)

        # Validation of rownumber passed by user.
        if ((myrowSelected < 0) or (myrowSelected > testDF.shape[0])):
            print('Value entered is out of range!!!!!!')
            myrowSelected = int(1)
            exceptionVar = 'Oops! You selected out of range value, getting prediction for default first record.'
        else:

            myrowSelected = myrowSelected + 1
            print('Value is correct!!!!!!')

        # Filtering the record from test data as per the rownumber passed by user.
        meat_test_df2 = pd.DataFrame(testDF.iloc[myrowSelected - 1:myrowSelected])

        meat_test_df = meat_test_df2.iloc[:, :5]
        actualLabel = meat_test_df2.iloc[:, -1]

        meat_test_df['type'] = actualLabel

        plsmlmodel = pickle.load(open(myModel, 'rb'))
        testDF_noTYPE = testDF.iloc[myrowSelected - 1:myrowSelected].copy()
        testDF_noTYPE = testDF_noTYPE.drop("type", axis='columns')

        print(testDF_noTYPE)
        print('Before prediction, plsmlmodel: ', plsmlmodel)

        # Calling predict using the model PLSnew.pkl
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
    # Exception handelling for file not found.
    except FileNotFoundError:

        return render_template('meatType_prediction.html', error_message=f' {error_message}', error_codeData=1)

    # print(prediction)

    except ValueError:
        return render_template('meatType_prediction.html', error_message=f' Exception! ,  Please enter numeric value For the field : "Enter Row Number " ', error_codeData=1)
    # returning the single row used for prediction and the other processed information.
    return render_template('meatType_prediction.html', meataccuracy=MLaccuracy, exceptionVar=exceptionVar,
                           meatdata=[meat_test_df.to_html()], columns=[''],
                           Mlresult=f' ML Predicted Meat Type: {predictedLabel}', error_codeData=0, focus_div=focus_div)


@meatApp.route('/MeatDF', methods=("POST", "GET"))
def DFMeat_tab():
    # Called by Button : Meat Data , page: meatType_prediction.html
    # Method is created to read the full test data and show it in User interface.

    error_codeData = 0

    # Exception handelling if the actual data set exists in the system.
    if os.path.isfile('static/upload/meats.csv'):
        error_message = "Exception! , Please hit button 'Process Meat Data'  and then Retry!  "
    else:
        error_message = "Exception! , Please first upload the data : dataset meats.csv and then re process!  "

    try:
        MeatDf = pd.read_csv('static/upload/meatTest_PLS.csv', index_col=0)

        # Dataframe has 1050 columns, to show in UI , taking few columns only.
        meat_user_viewDF = MeatDf.iloc[:, :5]
        meat_user_viewDF['type'] = MeatDf.iloc[:, -1]

        pass
    # Exception handelling for file not found.
    except FileNotFoundError:

        return render_template('meatType_prediction.html', error_message=f' {error_message}', error_codeData=1)

    # returning the all row used for prediction and the other processed information.
    return render_template('meatType_prediction.html', columns=[''], meatdata=[meat_user_viewDF.to_html()])


@meatApp.route('/upload', methods=['GET', 'POST'])
def upload():
    # Called by Button : Upload Meat Data , page : meatType_prediction.html
    # This method uploads the actual data set provided by the user in upload folder.
    error_code = 0
    if request.method == 'GET':

        print('request method is GET')

    else:

        error_message = 'Exception! , Please browse the file meats.csv and then Retry!'
        try:

            print('request method is POST')
            meatfile = request.files['file']
            meatfilename = meatfile.filename
            print('meatfilename  :  ', meatfilename)
            print(f" upload folder path is ) {meatApp.config['data_upload']}")

            if meatfilename is None:
                return render_template('meatType_prediction.html', error_message=f'Please browse file Meats.csv then Retry! ', error_code=2)
            meatfile.save(os.path.join(meatApp.config['data_upload'], meatfilename))

            pass

        # Exception handelling for file not found.
        except FileNotFoundError:

            return render_template('meatType_prediction.html', error_message=f' {error_message}', error_code=2)
        # returning  success message after upload.
        return render_template('meatType_prediction.html', result='DataUploaded successfully.')


@meatApp.route('/processML', methods=['GET', 'POST'])
def processML():
    # Called by Button :Process Meat Data , page: meatType_prediction.html
    # Method is created to run split train and test data set.
    # Create the ML model and save it in upload folder.
    # Return the message

    # Intialization of error code
    error_code = 0

    # Checking if the actual data set exists in upload folder.
    if os.path.isfile('static/upload/meats.csv'):
        error_message = "Exception! , Please hit button 'Process Meat Data'  and then Retry!  "
    else:
        error_message = "Exception! , Please first upload the data : dataset meats.csv and then re process!  "

    try:
        meatDF = pd.read_csv("static/upload/meats.csv")
        # Getting meat data without Label
        meat_nolabel_df = meatDF.iloc[:, 1:1050].copy()

        # Getting the column with just label
        meat_label = meatDF.iloc[:, -1].copy()

        # Splitting the Data set
        meatdf_TRAIN_X, meatdf_TEST_X, meatlabel_trainY, meatlabel_testY = train_test_split(meat_nolabel_df, meat_label,
                                                                                            stratify=meat_label,
                                                                                            test_size=0.2,
                                                                                            random_state=1121218)

        n = len(meatdf_TRAIN_X)

        # Using the 10-fold and shuffle as True
        kfold_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)
        mse = []

        meatlabel_trainY = meatlabel_trainY.reset_index(drop=True)

        # PLS uses the continious data, so converting the categorical data
        Meatlabel_encoder = LabelEncoder()
        meatlabel_trainY_F = Meatlabel_encoder.fit_transform(meatlabel_trainY)

        # Passing the n_components as 14
        algo = PLSRegression(n_components=14)

        meatlabel_trainY = meatlabel_trainY.reset_index(drop=True)
        meatlabel_trainY_F = pd.get_dummies(meatlabel_trainY)

        meatdf_TRAIN_X = meatdf_TRAIN_X.reset_index(drop=True)

        # Training the model
        algo.fit(meatdf_TRAIN_X, meatlabel_trainY_F)

        # Downloading the model
        mklFile = open('static/upload/PLSnew.pkl', 'wb')
        pickle.dump(algo, mklFile)
        mklFile.close()

        ##Downloding the test data for UI
        testDF = meatdf_TEST_X.copy().reset_index(drop=True)
        testDF['type'] = meatlabel_testY.reset_index(drop=True)
        testDF.insert(0, '', range(len(testDF)))
        testDF.to_csv(r'static/upload/meatTest_PLS.csv', index=False)

        # Calling the predict
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


    # Exception handelling for file not found.
    except FileNotFoundError:

        return render_template('meatType_prediction.html', error_message=f' {error_message}', error_code=1)

    # returning the ML accuracy
    return render_template('meatType_prediction.html', MLAccuracy=f' ML Accuracy is : {Accuracy}')


if __name__ == "__main__":
    meatApp.run(host = '0.0.0.0', port = 80 , debug = True)
