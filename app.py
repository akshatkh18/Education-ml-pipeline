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

    result = session.pop('result', None)
    form_data = session.pop('form_data', None)

    return render_template(
        'home.html',
        result=result,
        form_data=form_data
    )


@app.route('/predictdata', methods=['POST'])
def predict_datapoint():

    try:

        # Save form values so they remain after prediction
        form_data = request.form.to_dict()

        data = CustomData(
            gender=form_data['gender'],
            race_ethnicity=form_data['race_ethnicity'],
            parental_level_of_education=form_data['parental_level_of_education'],
            lunch=form_data['lunch'],
            test_preparation_course=form_data['test_preparation_course'],
            reading_score=float(form_data['reading_score']),
            writing_score=float(form_data['writing_score']),
        )

        pred_df = data.get_data_as_data_frame()

        result = predict_pipeline.predict(pred_df)[0]

        # Clamp between 0–100
        result = round(max(0, min(100, result)), 2)

        session['result'] = result
        session['form_data'] = form_data

        return redirect(url_for('home'))

    except Exception as e:
        return f"Error occurred: {e}"


if __name__ == "__main__":
    app.run(debug=True)
