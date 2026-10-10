"""Student performance application and Gunicorn entry point."""
import os
from flask import Flask, redirect, render_template, request, url_for
from src.pipeline.predict_pipeline import CATEGORIES, CustomData, PredictPipeline


def create_app(pipeline=None):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024
    app.extensions['predict_pipeline'] = pipeline if pipeline is not None else PredictPipeline()

    def page(form_data=None, result=None, errors=None, status=200):
        return render_template('home.html', categories=CATEGORIES,
                               form_data=form_data or {}, result=result,
                               errors=errors or {}), status

    @app.get('/')
    def home():
        return page()

    @app.get('/predictdata')
    def prediction_form():
        return redirect(url_for('home'))

    @app.post('/predictdata')
    def predict_datapoint():
        form_data = request.form.to_dict()
        errors = CustomData.validate(form_data)
        if errors:
            return page(form_data, errors=errors, status=400)
        try:
            data = CustomData(**{key: form_data[key] for key in CustomData.fields})
            prediction = app.extensions['predict_pipeline'].predict(data.get_data_as_data_frame())[0]
            result = round(float(max(0, min(100, prediction))), 2)
            return page(form_data, result=result)
        except Exception:
            app.logger.exception('Prediction failed')
            return page(form_data, errors={'general': 'Prediction is temporarily unavailable. Please try again.'}, status=503)

    @app.get('/healthz')
    def health():
        return {'status': 'ok'}

    @app.errorhandler(413)
    def too_large(error):
        return page(errors={'general': 'This request is too large. Please submit only the form fields.'}, status=413)

    @app.after_request
    def response_headers(response):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'same-origin'
        response.headers['Content-Security-Policy'] = "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; form-action 'self'; frame-ancestors 'none'; base-uri 'self'"
        return response

    return app


app = create_app()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', '5000')), debug=False)
