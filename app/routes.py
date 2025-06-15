from flask import Blueprint, render_template, request, jsonify
import pandas as pd
import numpy as np
import seaborn as sns

import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, KBinsDiscretizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score
from matplotlib import pyplot as plt
from datetime import datetime

main = Blueprint("main", __name__)


requests_history = []

# ------------------------------------------------------------------
# 1) LOAD AND PRE‑PROCESS THE DATASET ONCE WHEN APP STARTS
# ------------------------------------------------------------------
RAW_CSV = "/home/husenkh/mysite/loan.csv"

df = pd.read_csv(RAW_CSV)
df.columns = [
    "#", "LoanId", "Gender", "Married", "Dependents", "Education",
    "SelfEmployed", "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "LoanAmountTerm", "CreditHistory", "PropertyArea", "target"
]
df.drop(columns=["LoanId", "#"], inplace=True)

cat_cols = [
    "Gender", "Married", "Dependents", "SelfEmployed",
    "CreditHistory", "LoanAmountTerm"
]
for c in cat_cols:
    df[c].fillna(df[c].mode()[0], inplace=True)
df["LoanAmount"].fillna(df["LoanAmount"].mean(), inplace=True)

# Store and reuse label encoders
label_encoders = {}
for col in [
    "Gender", "Married", "Dependents", "Education",
    "LoanAmountTerm", "SelfEmployed", "CreditHistory", "PropertyArea"
]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

df["target"] = df["target"].map({"Y": 1, "N": 0})
cat_cols=[ "Gender", "Married", "Dependents", "Education",    "LoanAmountTerm", "SelfEmployed", "CreditHistory", "PropertyArea"]
num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]
for col in num_cols:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

disc_cols = ["ApplicantIncomeCat", "CoapplicantIncomeCat", "LoanAmountCat"]
kbd = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="kmeans")
df[disc_cols] = kbd.fit_transform(df[num_cols])

features = disc_cols + [
    "Gender", "Married", "Dependents", "Education",
    "LoanAmountTerm", "SelfEmployed", "CreditHistory", "PropertyArea"
]

# Save heatmap of correlations
plt.figure(figsize=(8,6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='inferno')
heatmap_path = 'static/images/correlation_heatmap.png'
# plt.savefig(heatmap_path)
plt.clf()

# Save boxplots of numerical columns
plt.figure(figsize=(12,4))
for i, col in enumerate(num_cols):
    plt.subplot(1, 3, i+1)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
boxplots_path = 'static/images/boxplots.png'
# plt.savefig(boxplots_path)
plt.clf()

# Countplots for categorical columns
plt.figure(figsize=(15,8))
for i, col in enumerate(cat_cols):
    plt.subplot(3, 3, i+1)
    sns.countplot(x=df[col])
    plt.title(f'Countplot of {col}')
countplots_path = 'static/images/countplots.png'
# plt.savefig(countplots_path)
plt.clf()

# Basic data info and head
info = str(df.info())
head_html = df.head().to_html(classes='table table-striped')

X = df[features]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=47
)

svc = SVC(probability=True)
param_grid = {"C": list(range(1, 100, 10)), "kernel": ["linear"]}
grid = GridSearchCV(svc, param_grid=param_grid, scoring="f1", cv=5)
grid.fit(X_train, y_train)
svm_model = grid.best_estimator_

y_pred = svm_model.predict(X_test)
print(
    f"[SVM] best C={grid.best_params_['C']}  "
    f"Accuracy={accuracy_score(y_test, y_pred):.3f}  "
    f"Precision={precision_score(y_test, y_pred):.3f}"
)

# ------------------------------------------------------------------
# 3) ROUTES
# ------------------------------------------------------------------
@main.route("/")
def home():
    return render_template("home.html")

@main.route("/requests")
def requests_page():
    return render_template("requests.html")

@main.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form.to_dict()

        # Encode categorical inputs
        row = {
            "ApplicantIncome": float(form["ApplicantIncome"]),
            "CoapplicantIncome": float(form["CoapplicantIncome"]),
            "LoanAmount": float(form["LoanAmount"]),
            "Gender": label_encoders["Gender"].transform([form["Gender"]])[0],
            "Married": label_encoders["Married"].transform([form["Married"]])[0],
            "Dependents": label_encoders["Dependents"].transform([form["Dependents"]])[0],
            "Education": label_encoders["Education"].transform([form["Education"]])[0],
            "SelfEmployed": label_encoders["SelfEmployed"].transform([form["SelfEmployed"]])[0],
            "LoanAmountTerm": label_encoders["LoanAmountTerm"].transform([form["LoanAmountTerm"]])[0],
            "CreditHistory": label_encoders["CreditHistory"].transform([form["CreditHistory"]])[0],
            "PropertyArea": label_encoders["PropertyArea"].transform([form["PropertyArea"]])[0],
        }

        inp_df = pd.DataFrame([row])
        inp_df[num_cols] = scaler.transform(inp_df[num_cols])
        inp_df[disc_cols] = kbd.transform(inp_df[num_cols])
        inp_df = inp_df[features]

        pred = svm_model.predict(inp_df)[0]
        prob = svm_model.predict_proba(inp_df)[0][1]

        # Save to history
        history_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "input": form,
            "prediction": "Approved" if pred else "Rejected",
            "probability": prob
        }
        requests_history.append(history_entry)

    except Exception as e:
        logging.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500

    res = f"Prediction: {'Approved ✅' if pred else 'Rejected ❌'}<br>Probability: {prob:.2f}"
    return render_template("result.html", result=res)


@main.route("/eda")
def eda():

    return render_template("eda.html")

@main.route("/evaluation")
def evaluation():
    return render_template("evaluation.html",
                           rows=df.shape[0],
                           cols=df.shape[1],
                           heatmap=heatmap_path,
                           boxplots=boxplots_path,
                           countplots=countplots_path)
                        #   ,
                        #   head_html=head_html

@main.route("/data-issue")
def data_issue():
    return render_template("issue.html")

@main.route("/history")
def history():
    return render_template("history.html", history=requests_history)

