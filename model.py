import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv("./dataset/The_Cancer_data_1500_V2.csv")  # Make sure 'Diagnosis' column is present

X = df.drop(['Diagnosis'], axis=1)
y = df['Diagnosis']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train models
logreg = LogisticRegression().fit(X_train_scaled, y_train)
rf = RandomForestClassifier(random_state=42).fit(X_train_scaled, y_train)
svm = SVC(probability=True).fit(X_train_scaled, y_train)

# Prediction function
def predict_majority(input_data: dict):
    df_input = pd.DataFrame([input_data])
    input_scaled = scaler.transform(df_input)

    pred_log = logreg.predict(input_scaled)[0]
    pred_rf = rf.predict(input_scaled)[0]
    pred_svm = svm.predict(input_scaled)[0]

    preds = [pred_log, pred_rf, pred_svm]
    final = max(set(preds), key=preds.count)

    return final, preds
