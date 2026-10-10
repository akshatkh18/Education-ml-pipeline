"""Validated inference using the original, trusted local artifacts."""
from pathlib import Path
import warnings
import dill
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

CATEGORIES = {
    'gender': ('female', 'male'),
    'race_ethnicity': ('group A', 'group B', 'group C', 'group D', 'group E'),
    'parental_level_of_education': ("associate's degree", "bachelor's degree", 'high school', "master's degree", 'some college', 'some high school'),
    'lunch': ('free/reduced', 'standard'),
    'test_preparation_course': ('completed', 'none'),
}
SCORES = ('reading_score', 'writing_score')
COLUMNS = ('gender', 'race/ethnicity', 'parental level of education', 'lunch',
           'test preparation course', 'reading score', 'writing score')
ARTIFACT_DIR = Path(__file__).resolve().parents[2] / 'artifacts'


class PredictPipeline:
    def __init__(self, artifact_dir=None):
        directory = Path(artifact_dir) if artifact_dir is not None else ARTIFACT_DIR
        with warnings.catch_warnings():
            warnings.simplefilter('error', InconsistentVersionWarning)
            with (directory / 'model.pkl').open('rb') as stream:
                self.model = dill.load(stream)
            with (directory / 'preprocessor.pkl').open('rb') as stream:
                self.preprocessor = dill.load(stream)
        sample = {key: values[0] for key, values in CATEGORIES.items()}
        sample.update(reading_score=70, writing_score=70)
        self.predict(CustomData(**sample).get_data_as_data_frame())

    def predict(self, features):
        if list(features.columns) != list(COLUMNS) or features.empty:
            raise ValueError('Expected a nonempty dataframe with the seven input columns.')
        for field, column in zip(CustomData.fields, COLUMNS):
            if field in CATEGORIES:
                if not features[column].isin(CATEGORIES[field]).all():
                    raise ValueError(f'Invalid {field}')
            else:
                values = pd.to_numeric(features[column], errors='coerce')
                if not (np.isfinite(values) & values.between(0, 100)).all():
                    raise ValueError(f'Invalid {field}')
        transformed = self.preprocessor.transform(features)
        if transformed.shape[1] != self.model.n_features_in_:
            raise ValueError('Model and preprocessor feature counts differ.')
        predictions = np.asarray(self.model.predict(transformed), dtype=float)
        if predictions.shape != (len(features),) or not np.isfinite(predictions).all():
            raise ValueError('Model returned invalid predictions.')
        return predictions


class CustomData:
    fields = (*CATEGORIES, *SCORES)

    @staticmethod
    def validate(values):
        errors = {}
        for field, choices in CATEGORIES.items():
            if values.get(field) not in choices:
                errors[field] = 'Choose one of the listed options.'
        for field in SCORES:
            try:
                score = float(values.get(field, ''))
                if not np.isfinite(score) or not 0 <= score <= 100:
                    raise ValueError
            except (ValueError, TypeError, OverflowError):
                errors[field] = 'Enter a number from 0 to 100.'
        return errors

    def __init__(self, gender, race_ethnicity, parental_level_of_education, lunch,
                 test_preparation_course, reading_score, writing_score):
        values = dict(zip(self.fields, (gender, race_ethnicity, parental_level_of_education,
                                       lunch, test_preparation_course, reading_score, writing_score)))
        if self.validate(values):
            raise ValueError('Invalid prediction input')
        for field, value in values.items():
            setattr(self, field, float(value) if field in SCORES else value)

    def get_data_as_data_frame(self):
        return pd.DataFrame([[getattr(self, field) for field in self.fields]], columns=COLUMNS)
