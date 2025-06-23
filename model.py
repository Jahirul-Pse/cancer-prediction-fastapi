import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Global variables
logreg = None
rf = None
svm = None
knn = None
gb = None
nb = None
scaler = None

def train_models():
    global logreg, rf, svm, knn, gb, nb, scaler

    # Load dataset
    df = pd.read_csv("./dataset/The_Cancer_data_1500_V2.csv")
    X = df.drop(['Diagnosis'], axis=1)
    y = df['Diagnosis']

    # Train/test split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train models
    logreg = LogisticRegression().fit(X_train_scaled, y_train)
    rf = RandomForestClassifier(random_state=42).fit(X_train_scaled, y_train)
    svm = SVC(probability=True).fit(X_train_scaled, y_train)
    knn = KNeighborsClassifier().fit(X_train_scaled, y_train)
    gb = GradientBoostingClassifier(random_state=42).fit(X_train_scaled, y_train)
    nb = GaussianNB().fit(X_train_scaled, y_train)

def predict_majority(input_data: dict):
    global logreg, rf, svm, knn, gb, nb, scaler

    df_input = pd.DataFrame([input_data])
    input_scaled = scaler.transform(df_input)

    pred_log = logreg.predict(input_scaled)[0]
    pred_rf = rf.predict(input_scaled)[0]
    pred_svm = svm.predict(input_scaled)[0]
    pred_knn = knn.predict(input_scaled)[0]
    pred_gb = gb.predict(input_scaled)[0]
    pred_nb = nb.predict(input_scaled)[0]

    votes = {
        "Logistic Regression": int(logreg.predict(input_scaled)[0]),
        "Random Forest": int(rf.predict(input_scaled)[0]),
        "SVM": int(svm.predict(input_scaled)[0]),
        "KNN": int(knn.predict(input_scaled)[0]),
        "Gradient Boosting": int(gb.predict(input_scaled)[0]),
        "Naive Bayes": int(nb.predict(input_scaled)[0])
    }
    # Majority vote
    predictions = list(votes.values())
    final = max(set(predictions), key=predictions.count)

    return final, votes
