#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RANDOM_SEED = 2
TEST_SIZE = 0.2


def featureSelectionWeakLearn(X_train, y_train, X_test, k):
    #Feature Selection using Information Gain
    from sklearn.feature_selection import mutual_info_classif
    mutual_info = mutual_info_classif(X_train, y_train, random_state=RANDOM_SEED)

    #select k columns with highest information gain
    sorted_mutual_info = sorted(mutual_info, reverse=True)
    threshold = sorted_mutual_info[k-1]
    columns = []

    for i in range(len(mutual_info)):
        if mutual_info[i] >= threshold:
            columns.append(i)  

    # print("Features count: ", len(columns))
    X_train = X_train[:, columns]
    X_test = X_test[:, columns]

    return X_train, X_test


#Preprocessing customer data
def preprocessing_dataset_1():
    dataframe = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    dataframe.replace(" ", np.nan, inplace = True)

    #drop customerID column
    dataframe.drop(['customerID'], axis=1, inplace=True)

    #change total charges to float
    to_float = ['TotalCharges']
    dataframe[to_float] = dataframe[to_float].astype(float)

    #change SeniorCitizen column to object
    to_object = ['SeniorCitizen']
    dataframe[to_object] = dataframe[to_object].astype(object)

    #target column is Churn
    X = dataframe.iloc[:, :-1].values
    y = dataframe.iloc[:, -1].values

    #label encoding for target column
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)


    #splitting the dataset into training and test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=TEST_SIZE, random_state=RANDOM_SEED)

    #replace TotalCharges missing values with mean of Train set
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X_train[:, 18:19])
    X_train[:, 18:19] = imputer.transform(X_train[:, 18:19])
    X_test[:, 18:19] = imputer.transform(X_test[:, 18:19])

    #Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    columns = [4, 17, 18]
    X_train[:, columns] = sc.fit_transform(X_train[:, columns])
    X_test[:, columns] = sc.transform(X_test[:, columns])

    #Label Encoding for the columns that have 2 values
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    columns = [0, 2, 3, 5, 15]
    for i in columns:
        X_train[:, i] = le.fit_transform(X_train[:, i])
        X_test[:, i] = le.transform(X_test[:, i])

    #OneHotEncoding for the columns that have more than 2 values
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    columns = [6, 7, 8, 9, 10, 11, 12, 13, 14, 16]
    ct = ColumnTransformer([('encoder', OneHotEncoder(), columns)], remainder='passthrough')
    X_train = ct.fit_transform(X_train)
    X_test = ct.transform(X_test)

    #change float64 to float32
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    #print shape of training and test set
    print("Dataset 1")
    print("X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)

    return X_train, X_test, y_train, y_test



def preprocessing_dataset_2():
    train_df = pd.read_csv('adult_train.csv')
    test_df = pd.read_csv('adult_test.csv')

    #we have to put a space before the question mark
    train_df.replace(' ?', np.nan, inplace=True)
    test_df.replace(' ?', np.nan, inplace=True)

    columns = ['workclass', 'occupation', 'native-country']
    #replace with most frequent value using sklearn
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    train_df[columns] = imputer.fit_transform(train_df[columns])
    test_df[columns] = imputer.fit_transform(test_df[columns])

    # if >50k then 1 else 0 in income column
    # the test set has a dot at the end of the value
    train_df['income'] = train_df['income'].map({' <=50K': 0, ' >50K': 1})
    test_df['income'] = test_df['income'].map({' <=50K.': 0, ' >50K.': 1})

    #drop the row with Holand-Netherlands in native-country column
    train_df.drop(train_df[train_df['native-country'] == ' Holand-Netherlands'].index, inplace=True)

    #fix the index
    train_df.reset_index(drop=True, inplace=True)

    #separate features and target
    X_train = train_df.drop('income', axis=1)
    y_train = train_df['income']
    X_test = test_df.drop('income', axis=1)
    y_test = test_df['income']

    #print dimension of X_train and X_test
    print("X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)



    #label encoding
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    key = ['sex']
    for column in key:
        X_train[column] = le.fit_transform(X_train[column])
        X_test[column] = le.transform(X_test[column])

    #convert to numpy array
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()


    #feature scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    columns = [0,2,4,10,11,12]
    X_train[:, columns] = sc.fit_transform(X_train[:, columns])
    X_test[:, columns] = sc.transform(X_test[:, columns])


    #One hot encoding using pandas
    train_df = pd.DataFrame(X_train)
    test_df = pd.DataFrame(X_test)

    #get_dummies for categorical columns
    columns = [1,3,5,6,7,8,13]
    train_df = pd.get_dummies(train_df, columns=columns)
    test_df = pd.get_dummies(test_df, columns=columns)

    X_train = train_df.to_numpy()
    X_test = test_df.to_numpy()


    #change float64 to float32
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    print(X_test.shape)
    print(X_train.shape)

    return X_train, X_test, y_train, y_test

def preprocessing_dataset_3(useSmaller = False):
    df = pd.read_csv('creditcard.csv')


    if useSmaller:
        df_class0 = df[df['Class'] == 0]
        df_class1 = df[df['Class'] == 1]
        #use 20000 random samples from class 0
        df_class0 = df_class0.sample(n=20000, random_state=RANDOM_SEED)
        #combine the two classes into one dataframe
        df = pd.concat([df_class0, df_class1])

    #split into training and test sets
    from sklearn.model_selection import train_test_split
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    #Feature scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    #change float64 to float32
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)


    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    #print shape of training and test set
    print("Dataset 3")
    print("X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)

    return X_train, X_test, y_train, y_test

#Logistic Regression using sigmoid function
class LogisticRegression:
    def __init__(self, learning_rate=0.05, iterations=2000):
        self.learning_rate = learning_rate
        self.iterations = iterations

    #sigmoid(z) = e^z / (1 + e^z)
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def gradient_descent(self, X, h, y):
        return np.dot(X.T, (y-h)) / y.shape[0]

    #cost function: J(theta) = -1/m * summation(yi * log(h(xi)) + (1 - yi) * log(1 - h(xi)))
    def logistic_error(self, h, y):
        error = 0
        for i in range(len(y)):
            error += y[i] * np.log(h[i]) + (1 - y[i]) * np.log(1 - h[i])
        error = -error / len(y)
        return error
        

    def train(self, X, y, weakLearn = False):

        #initialize theta
        samples_count, features_count = X.shape
        self.theta = np.zeros((features_count, 1))

        #training the model
        divide = self.iterations // 4
        for i in range(self.iterations):
            if i == divide:
                self.learning_rate = self.learning_rate / 10
            elif i == divide * 2:
                self.learning_rate = self.learning_rate
            elif i == divide * 3:
                self.learning_rate = self.learning_rate / 10
            

            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = self.gradient_descent(X, h, y)
            self.theta += self.learning_rate * gradient
            if weakLearn:
                error = self.logistic_error(h, y)
                if error < 0.05:
                    print("Error: ", error)
                    print("Iterations: ", i)
                    break

        #return theta
        return self.theta

    def predict(self, X):
        z = np.dot(X, self.theta)
        h = self.sigmoid(z)
        # return 1 for h >= 0.5 else return 0
        return np.where(h >= 0.5, 1, 0)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return (y_pred == y).mean()
    
    def confusion_matrix(self, X, y):
        y_pred = self.predict(X)
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(len(y)):
            if y[i] == 1 and y_pred[i] == 1:
                TP += 1
            elif y[i] == 0 and y_pred[i] == 0:
                TN += 1
            elif y[i] == 0 and y_pred[i] == 1:
                FP += 1
            else:
                FN += 1
        return TP, TN, FP, FN
    
    #accuracy, sensitivity, specificity, precision, false discovery rate, f1 score
    def performance_metrics(self, X, y):
        TP, TN, FP, FN = self.confusion_matrix(X, y)
        print(TP, "," , TN, ",", FP, ",", FN)

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        precision = TP / (TP + FP)
        fdr = FP / (FP + TP)
        f1_score = 2 * precision * sensitivity / (precision + sensitivity)
        return accuracy, sensitivity, specificity, precision, fdr, f1_score


#Addaptive Boosting using Logistic Regression

class AdaBoost:
    def __init__(self, n_estimators=5, weakLearn = False):
        self.n_estimators = n_estimators
        self.weakLearn = weakLearn
        self.h = []
        self.z = []

    def resample(self, X, y, w):
        #resample examples according to weights
        np.random.seed(RANDOM_SEED)
        sample = np.random.choice(np.arange(len(X)), size=X.shape[0], replace=True, p=w)
        return X[sample, :], y[sample]
    
    #function adaboosting takes examples, number of estimators as input and returns a weighted majority hypothesis
    def adaboosting(self, X, y, L, n_estimators):
        w = np.ones(X.shape[0]) / X.shape[0]
        h = [] #a vector of k hypothesis
        z = [] #a vector of k hyoithesis weights

        for i in range(n_estimators):
            #resample examples according to weights
            X_resampled, y_resampled = self.resample(X, y, w)

            #train a hypothesis
            model = L(iterations=200)
            theta = model.train(X_resampled, y_resampled, weakLearn=self.weakLearn)

            #predict on training set
            y_pred = model.predict(X)

            #calculate error
            error = 0
            for j in range(len(y)):
                if y_pred[j] != y[j]:
                    error += w[j]
            
            if error > 0.5:
                continue

            
            self.h.append(model)

            #calculate hypothesis weight
            for j in range(len(y)):
                if y_pred[j] == y[j]:
                    w[j] = w[j] * error / (1 - error)

            #normalize weights
            w = w / w.sum()

            #update hypothesis weight
            self.z.append(np.log((1 - error) / error))
        
        
    
    #function weighted_majority takes a vector of hypothesis models, a vector of hypothesis weights and X as input and returns a weighted majority hypothesis

    def weighted_majority(self, X):
        y_pred = []
        h = self.h
        z = self.z

        print("Number of Hypothesis: ", len(h))
        #take summation of weighted predictions
        for i in range(len(h)):
            y_pred.append(h[i].predict(X))
            #change 0 to -1
            y_pred[i] = np.where(y_pred[i] == 0, -1, y_pred[i])

           

        y_pred = np.array(y_pred)
        #calculate weighted majority hypothesis
        weighted_majority = np.dot(y_pred.T, z)

        #print dimension of weighted majority
        print("Weighted Majority Dimension: ", weighted_majority.shape)
        #return a n x 1 vector of predictions
        predictions = np.where(weighted_majority >= 0, 1, 0)
        
        #reshape predictions to a 1D array
        predictions = predictions.reshape(-1)
        return predictions

    
    def confusion_matrix(self, X, y):
        y_pred = self.weighted_majority(X)
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(len(y)):
            if y[i] == 1 and y_pred[i] == 1:
                TP += 1
            elif y[i] == 0 and y_pred[i] == 0:
                TN += 1
            elif y[i] == 0 and y_pred[i] == 1:
                FP += 1
            else:
                FN += 1
        self.TP = TP
        self.TN = TN
        self.FP = FP
        self.FN = FN
    
    def performance_metrics(self):
        accuracy = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
        sensitivity = self.TP / (self.TP + self.FN)
        specificity = self.TN / (self.TN + self.FP)
        precision = self.TP / (self.TP + self.FP)
        fdr = self.FP / (self.FP + self.TP)
        f1_score = 2 * precision * sensitivity / (precision + sensitivity)
        return accuracy, sensitivity, specificity, precision, fdr, f1_score


def run_logistic_regression(X_train, X_test, y_train, y_test, weakLearn = False, k = 10):
    if weakLearn:
        X_train, X_test = featureSelectionWeakLearn(X_train, y_train, X_test, k)
    
    #adding bias
    bias = np.ones((X_train.shape[0], 1))
    X_train = np.concatenate((bias, X_train), axis=1)
    bias = np.ones((X_test.shape[0], 1))
    X_test = np.concatenate((bias, X_test), axis=1)

    #prin dimension of X_train and X_test
    print("X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)

    #make y_train and y_test 2D arrays
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    model = LogisticRegression(iterations=20000)
    model.train(X_train, y_train)
    accuracy, sensitivity, specificity, precision, fdr, f1_score = model.performance_metrics(X_train, y_train)
    print("Training Set")
    print("Accuracy: ", accuracy)
    print("Sensitivity: ", sensitivity)
    print("Specificity: ", specificity)
    print("Precision: ", precision)
    print("False Discovery Rate: ", fdr)
    print("F1 Score: ", f1_score)
    accuracy, sensitivity, specificity, precision, fdr, f1_score = model.performance_metrics(X_test, y_test)
    print("Test Set")
    print("Accuracy: ", accuracy)
    print("Sensitivity: ", sensitivity)
    print("Specificity: ", specificity)
    print("Precision: ", precision)
    print("False Discovery Rate: ", fdr)
    print("F1 Score: ", f1_score)
    print()


def run_adaboost(X_train, X_test, y_train, y_test, k, weakLearn = False, features = 10):

    if weakLearn:
        X_train, X_test = featureSelectionWeakLearn(X_train, y_train, X_test, features)
    
    #adding bias
    bias = np.ones((X_train.shape[0], 1))
    X_train = np.concatenate((bias, X_train), axis=1)
    bias = np.ones((X_test.shape[0], 1))
    X_test = np.concatenate((bias, X_test), axis=1)

    #make y_train and y_test 2D arrays
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    model = AdaBoost(n_estimators=k)
    model.adaboosting(X_train, y_train, LogisticRegression, k)
    model.confusion_matrix(X_train, y_train)
    accuracy, sensitivity, specificity, precision, fdr, f1_score = model.performance_metrics()
    print("Train Set in AdaBoost")
    print("Accuracy: ", accuracy)
    print("Sensitivity: ", sensitivity)
    print("Specificity: ", specificity)
    print("Precision: ", precision)
    print("False Discovery Rate: ", fdr)
    print("F1 Score: ", f1_score)
    print()

    model.confusion_matrix(X_test, y_test)
    accuracy, sensitivity, specificity, precision, fdr, f1_score = model.performance_metrics()
    print("Test Set in AdaBoost")
    print("Accuracy: ", accuracy)
    print("Sensitivity: ", sensitivity)
    print("Specificity: ", specificity)
    print("Precision: ", precision)
    print("False Discovery Rate: ", fdr)
    print("F1 Score: ", f1_score)
    print("\n\n")
    print("--------------------------------------------------")
    print("\n\n")




#open file to write results
file = open("result.txt", "w")
#bind file to standard output
import sys
sys.stdout = file


def main():
    data1_features = 30
    # run_logistic_regression(*preprocessing_dataset_1(), weakLearn=False)
    run_adaboost(*preprocessing_dataset_1(), 5, weakLearn=True, features=data1_features)
    run_adaboost(*preprocessing_dataset_1(), 10, weakLearn=True, features=data1_features)
    run_adaboost(*preprocessing_dataset_1(), 15, weakLearn=True, features=data1_features)
    run_adaboost(*preprocessing_dataset_1(), 20, weakLearn=True, features=data1_features)

    data2_features = 50
    # run_logistic_regression(*preprocessing_dataset_2())
    # run_adaboost(*preprocessing_dataset_2(), 5, weakLearn=True, features=data2_features)
    # run_adaboost(*preprocessing_dataset_2(), 10, weakLearn=True, features=data2_features)
    # run_adaboost(*preprocessing_dataset_2(), 15, weakLearn=True, features=data2_features)
    # run_adaboost(*preprocessing_dataset_2(), 20, weakLearn=True, features=data2_features)

    data3_features = 20
    # run_logistic_regression(*preprocessing_dataset_3(useSmaller = True))
    # run_adaboost(*preprocessing_dataset_3(useSmaller = True), 5, weakLearn=True, features=data3_features)
    # run_adaboost(*preprocessing_dataset_3(useSmaller = True), 10, weakLearn=True, features=data3_features)
    # run_adaboost(*preprocessing_dataset_3(useSmaller = True), 15, weakLearn=True, features=data3_features)
    # run_adaboost(*preprocessing_dataset_3(useSmaller = True), 20, weakLearn=True, features=data3_features)

if __name__ == "__main__":
    main()
