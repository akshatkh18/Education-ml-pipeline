"""Run with python -m unittest discover -s tests -v."""
import hashlib
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from werkzeug.serving import make_server
from app import app, create_app
from src.pipeline.predict_pipeline import ARTIFACT_DIR, CATEGORIES, CustomData, PredictPipeline

SAMPLE = dict(gender='male', race_ethnicity='group C',
              parental_level_of_education="bachelor's degree", lunch='standard',
              test_preparation_course='completed', reading_score='75', writing_score='80')


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = PredictPipeline()

    def test_original_artifacts_unchanged(self):
        for name, digest in {
            'model.pkl': 'dee62c8a49a143086b38085eac598a052b73af05bbc101899f9ff03703b63f53',
            'preprocessor.pkl': 'f07b6f4c0a63731b2523f0005bfcf15394ed592d3be318bec265f2f904c655af',
        }.items():
            self.assertEqual(hashlib.sha256((ARTIFACT_DIR / name).read_bytes()).hexdigest(), digest)

    def test_saved_test_dataset(self):
        data = pd.read_csv(ARTIFACT_DIR / 'test.csv')
        inputs = data.drop(columns='math score')
        result = self.pipeline.predict(inputs)
        self.assertEqual(result.shape, (200,))
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(self.pipeline.preprocessor.transform(inputs).shape, (200, 19))
        self.assertGreater(r2_score(data['math score'], result), .8)
        print(f'\nSaved test set: R2={r2_score(data["math score"], result):.6f}, MAE={mean_absolute_error(data["math score"], result):.6f}')

    def test_all_category_options_and_score_boundaries(self):
        for field, choices in CATEGORIES.items():
            for value in choices:
                for score in (0, 75.5, 100):
                    with self.subTest(field=field, value=value, score=score):
                        data = {**SAMPLE, field: value, 'reading_score': score, 'writing_score': score}
                        self.assertTrue(np.isfinite(self.pipeline.predict(CustomData(**data).get_data_as_data_frame())).all())

    def test_artifact_paths_independent_of_cwd(self):
        import os
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                PredictPipeline()
            finally:
                os.chdir(original)

    def test_missing_and_corrupt_artifacts_fail_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                PredictPipeline(directory)
            Path(directory, 'model.pkl').write_bytes(b'broken')
            with self.assertRaises(Exception):
                PredictPipeline(directory)

    def test_pipeline_rejects_bad_values_and_nonfinite_output(self):
        features = CustomData(**SAMPLE).get_data_as_data_frame()
        bad = features.copy()
        bad['reading score'] = np.nan
        with self.assertRaises(ValueError):
            self.pipeline.predict(bad)
        with patch.object(self.pipeline.model, 'predict', return_value=np.array([np.nan])):
            with self.assertRaises(ValueError):
                self.pipeline.predict(features)


class WebTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_home_health_static_and_get_prediction(self):
        for path in ('/', '/healthz', '/static/app.css', '/static/app.js'):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                response.close()
        self.assertEqual(self.client.get('/predictdata').status_code, 302)
        self.assertEqual(self.client.get('/missing').status_code, 404)

    def test_prediction_matches_pipeline_and_retains_inputs(self):
        response = self.client.post('/predictdata', data=SAMPLE)
        expected = app.extensions['predict_pipeline'].predict(CustomData(**SAMPLE).get_data_as_data_frame())[0]
        self.assertEqual(response.status_code, 200)
        self.assertIn(f'{expected:.2f}'.encode(), response.data)
        self.assertIn(b'value="75"', response.data)
        self.assertNotIn('Set-Cookie', response.headers)
        self.assertEqual(response.headers['Cache-Control'], 'no-store')

    def test_invalid_scores(self):
        for field in ('reading_score', 'writing_score'):
            for value in ('', '-1', '101', 'NaN', 'inf', '-inf', '1e9999', 'abc'):
                with self.subTest(field=field, value=value):
                    response = self.client.post('/predictdata', data={**SAMPLE, field: value})
                    self.assertEqual(response.status_code, 400)
                    self.assertIn(b'Enter a number from 0 to 100.', response.data)

    def test_missing_fields_and_unknown_categories(self):
        for field in SAMPLE:
            data = SAMPLE.copy()
            del data[field]
            with self.subTest(field=field):
                self.assertEqual(self.client.post('/predictdata', data=data).status_code, 400)
        for field in CATEGORIES:
            self.assertEqual(self.client.post('/predictdata', data={**SAMPLE, field: 'unknown'}).status_code, 400)

    def test_zero_and_clamped_scores_are_visible(self):
        for raw, shown in ((0, '0.00'), (-10, '0.00'), (120, '100.00')):
            with patch.object(app.extensions['predict_pipeline'], 'predict', return_value=[raw]):
                response = self.client.post('/predictdata', data=SAMPLE)
                self.assertIn(f'id="prediction-value">{shown}'.encode(), response.data)

    def test_service_failure_is_generic(self):
        with patch.object(app.extensions['predict_pipeline'], 'predict', side_effect=RuntimeError('SECRET_INTERNAL_PATH')):
            with self.assertLogs(app.logger, level='ERROR'):
                response = self.client.post('/predictdata', data=SAMPLE)
        self.assertEqual(response.status_code, 503)
        self.assertNotIn(b'SECRET_INTERNAL_PATH', response.data)
        self.assertIn(b'temporarily unavailable', response.data)

    def test_oversized_and_nonform_requests(self):
        self.assertEqual(self.client.post('/predictdata', data={'extra': 'x' * 20000}).status_code, 413)
        self.assertEqual(self.client.post('/predictdata', json=SAMPLE).status_code, 400)

    def test_reflected_values_are_escaped(self):
        response = self.client.post('/predictdata', data={**SAMPLE, 'reading_score': '"><script>alert(1)</script>'})
        self.assertNotIn(b'<script>alert(1)</script>', response.data)

    def test_independent_app_instances_need_no_shared_session(self):
        second = create_app(app.extensions['predict_pipeline']).test_client()
        self.assertEqual(second.post('/predictdata', data=SAMPLE).status_code, 200)

    def test_actual_http_requests(self):
        server = make_server('127.0.0.1', 0, app)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base + '/', timeout=10) as response:
                self.assertEqual(response.status, 200)
            with urlopen(Request(base + '/predictdata', data=urlencode(SAMPLE).encode()), timeout=10) as response:
                self.assertEqual(response.status, 200)
                self.assertIn(b'id="prediction-value"', response.read())
            with self.assertRaises(HTTPError) as caught:
                urlopen(Request(base + '/predictdata', data=b'reading_score=NaN'), timeout=10)
            self.assertEqual(caught.exception.code, 400)
            caught.exception.close()
            with urlopen(base + '/healthz', timeout=10) as response:
                self.assertEqual(response.status, 200)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()


if __name__ == '__main__':
    unittest.main()
