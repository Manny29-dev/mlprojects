import os
import sys
import pickle
import dill

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(os.path.abspath(file_path))
        os.makedirs(dir_path, exist_ok=True)

        # Use dill for more robust serialization of complex objects
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)