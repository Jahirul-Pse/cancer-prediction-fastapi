import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Add this global dictionary
accuracies = {}

# Global variables
logreg = None
rf = None
svm = None
knn = None
nb = None
scaler = None

def train_models():
    global logreg, rf, svm, knn, nb, scaler, accuracies

    df = pd.read_csv("./dataset/The_Cancer_data_1500_V2.csv")
    X = df.drop(['Diagnosis'], axis=1)
    y = df['Diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    logreg = LogisticRegression().fit(X_train_scaled, y_train)
    rf = RandomForestClassifier(random_state=42).fit(X_train_scaled, y_train)
    svm = SVC(probability=True).fit(X_train_scaled, y_train)
    knn = KNeighborsClassifier().fit(X_train_scaled, y_train)
    nb = GaussianNB().fit(X_train_scaled, y_train)

    # Compute accuracies
    accuracies = {
        "Logistic Regression": accuracy_score(y_test, logreg.predict(X_test_scaled)),
        "Random Forest": accuracy_score(y_test, rf.predict(X_test_scaled)),
        "SVM": accuracy_score(y_test, svm.predict(X_test_scaled)),
        "KNN": accuracy_score(y_test, knn.predict(X_test_scaled)),
        "Naive Bayes": accuracy_score(y_test, nb.predict(X_test_scaled))
    }

def get_model_accuracies():
    return accuracies

def predict_majority(input_data: dict):
    global logreg, rf, svm, knn, nb, scaler

    df_input = pd.DataFrame([input_data])
    input_scaled = scaler.transform(df_input)

    pred_log = logreg.predict(input_scaled)[0]
    pred_rf = rf.predict(input_scaled)[0]
    pred_svm = svm.predict(input_scaled)[0]
    pred_knn = knn.predict(input_scaled)[0]
    pred_nb = nb.predict(input_scaled)[0]

    votes = {
        "Logistic Regression": int(logreg.predict(input_scaled)[0]),
        "Random Forest": int(rf.predict(input_scaled)[0]),
        "SVM": int(svm.predict(input_scaled)[0]),
        "KNN": int(knn.predict(input_scaled)[0]),
        "Naive Bayes": int(nb.predict(input_scaled)[0])
    }
    # Majority vote
    predictions = list(votes.values())
    final = max(set(predictions), key=predictions.count)

    return final, votes
