from flask import Flask, request, render_template, redirect, url_for, session
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import os

app = Flask(__name__)

# Needed for session
app.secret_key = os.urandom(24)

# Load pipeline once (VERY IMPORTANT)
predict_pipeline = PredictPipeline()


@app.route('/')
def home():

    # pop removes value after showing once
    result = session.pop('result', None)

    return render_template('home.html', result=result)


@app.route('/predictdata', methods=['POST'])
def predict_datapoint():

    try:

        data = CustomData(
            gender=request.form['gender'],
            race_ethnicity=request.form['race_ethnicity'],
            parental_level_of_education=request.form['parental_level_of_education'],
            lunch=request.form['lunch'],
            test_preparation_course=request.form['test_preparation_course'],
            reading_score=float(request.form['reading_score']),
            writing_score=float(request.form['writing_score']),
        )

        pred_df = data.get_data_as_data_frame()

        result = predict_pipeline.predict(pred_df)[0]

        # Clamp result between 0–100
        result = round(max(0, min(100, result)), 2)

        # Store in session
        session['result'] = result

        return redirect(url_for('home'))

    except Exception as e:
        return f"Error occurred: {e}"


if __name__ == "__main__":
    app.run(debug=True)
