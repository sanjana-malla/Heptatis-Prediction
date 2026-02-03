# from flask import Flask, render_template, request
# import pandas as pd
# import joblib

# app = Flask(__name__)


# model = joblib.load("hepatitis_xgb_model.pkl")
# imputer = joblib.load("hepatitis_imputer.pkl")

# @app.route("/", methods=["GET", "POST"])
# def index():
#     result = None
#     advice = None
#     color = None

#     if request.method == "POST":
#         try:
            
#             data = {
#                 'age': [float(request.form['age'])],
#                 'sex': [1 if request.form['sex'].strip().lower() == 'male' else 0],
#                 'steroid': [int(request.form['steroid'])],
#                 'antivirals': [int(request.form['antivirals'])],
#                 'fatigue': [int(request.form['fatigue'])],
#                 'anorexia': [int(request.form['anorexia'])],
#                 'liver_big': [int(request.form['liver_big'])],
#                 'liver_firm': [int(request.form['liver_firm'])],
#                 'bilirubin': [float(request.form['bilirubin'])],
#                 'albumin': [float(request.form['albumin'])]
#             }

#             df_input = pd.DataFrame(data)
#             df_input = pd.DataFrame(imputer.transform(df_input), columns=df_input.columns)

        
#             prediction = int(model.predict(df_input)[0])
#             probability = model.predict_proba(df_input)[0][prediction] * 100

        
#             if prediction == 1:
#                 result = f"⚠️ Risk / Hepatitis (Confidence: {probability:.2f}%)"
#                 color = "red"
#                 advice = (
#                     "⚠️ You may be at risk of Hepatitis.<br>"
#                     "➡️ Consult a hepatologist or physician.<br>"
#                     "➡️ Get liver function tests (LFTs).<br>"
#                     "➡️ Avoid alcohol and fatty foods.<br>"
#                     "➡️ Follow prescribed medication and regular check-ups."
#                 )
#             else:
#                 result = f"✅ Healthy / No Risk (Confidence: {probability:.2f}%)"
#                 color = "green"
#                 advice = (
#                     "✅ You appear to be healthy.<br>"
#                     "➡️ Maintain a balanced diet and exercise regularly.<br>"
#                     "➡️ Avoid alcohol and get periodic liver check-ups."
#                 )

#         except Exception as e:
#             result = "Error"
#             color = "orange"
#             advice = f"An error occurred: {str(e)}"

#     return render_template("index.html", result=result, advice=advice, color=color)

# if __name__ == "__main__":
#     app.run(debug=True)



from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("hepatitis_xgb_model.pkl")
imputer = joblib.load("hepatitis_imputer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    advice = None
    color = None
    data = None
    probabilities = None 

    if request.method == "POST":
        try:
            
            data = {
                'age': [float(request.form['age'])],
                'sex': [1 if request.form['sex'].strip().lower() == 'male' else 0],
                'steroid': [int(request.form['steroid'])],
                'antivirals': [int(request.form['antivirals'])],
                'fatigue': [int(request.form['fatigue'])],
                'anorexia': [int(request.form['anorexia'])],
                'liver_big': [int(request.form['liver_big'])],
                'liver_firm': [int(request.form['liver_firm'])],
                'bilirubin': [float(request.form['bilirubin'])],
                'albumin': [float(request.form['albumin'])]
            }

            df_input = pd.DataFrame(data)
            df_input = pd.DataFrame(imputer.transform(df_input), columns=df_input.columns)

        
            prediction = int(model.predict(df_input)[0])
            probabilities = model.predict_proba(df_input)[0].tolist()
            probability = probabilities[prediction] * 100

            if prediction == 1:
                result = f"⚠️ Risk / Hepatitis (Confidence: {probability:.2f}%)"
                color = "red"
                advice = (
                    "⚠️ You may be at risk of Hepatitis.<br>"
                    "➡️ Consult a hepatologist or physician.<br>"
                    "➡️ Get liver function tests (LFTs).<br>"
                    "➡️ Avoid alcohol and fatty foods.<br>"
                    "➡️ Follow prescribed medication and regular check-ups."
                )
            else:
                result = f"✅ Healthy / No Risk (Confidence: {probability:.2f}%)"
                color = "green"
                advice = (
                    "✅ You appear to be healthy.<br>"
                    "➡️ Maintain a balanced diet and exercise regularly.<br>"
                    "➡️ Avoid alcohol and get periodic liver check-ups."
                )

        except Exception as e:
            result = "Error"
            color = "orange"
            advice = f"An error occurred: {str(e)}"


    return render_template(
        "index.html",
        result=result,
        advice=advice,
        color=color,
        input_data=data,
        probabilities=probabilities
    )

if __name__ == "__main__":
    app.run(debug=True)
