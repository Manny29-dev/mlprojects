import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: float,
        writing_score: float,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            data = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "writing score": [self.writing_score],
                "reading score": [self.reading_score],
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e, sys)

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path = 'artifact/preprocessor.pkl'
            model_path = 'artifact/model.pkl'

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            raise CustomException(e, sys)
        
    def get_data_as_dataframe(self,features):
        try:
            custom_data_input_dict = {
                "gender": [features['gender']],
                "race/ethnicity": [features['race/ethnicity']],
                "parental level of education": [features['parental level of education']],
                "lunch": [features['lunch']],
                "test preparation course": [features['test preparation course']],
                "writing score": [features['writing score']],
                "reading score": [features['reading score']]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
        