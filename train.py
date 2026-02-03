import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import joblib


df = pd.read_csv("hepatitis.csv")


features = ['age','sex','steroid','antivirals','fatigue','anorexia','liver_big','liver_firm','bilirubin','albumin']
X = df[features]


X['sex'] = X['sex'].apply(lambda x: 1 if str(x).lower() == 'male' else 0)
bool_cols = ['steroid','antivirals','fatigue','anorexia','liver_big','liver_firm']
for col in bool_cols:
    X[col] = X[col].apply(lambda x: 1 if str(x).lower() == 'true' else 0)


imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)


y = df['class'].apply(lambda x: 1 if str(x).lower() == 'live' else 0)


joblib.dump(imputer, "hepatitis_imputer.pkl")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


live_count = (y == 1).sum()
risk_count = (y == 0).sum()
scale_pos_weight = risk_count / live_count if live_count != 0 else 1


model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight
)
model.fit(X_train, y_train)


joblib.dump(model, "hepatitis_xgb_model.pkl")
print("✅ XGBoost model trained and saved successfully!")
