import sys

def error_message_details(error, error_details:sys):
    _, _, exc = error_details.exc_info()
    filename = exc.tb_frame.f_code.co_filename
    line_number = exc.tb_lineno
    error_message = f"Error occurred in script: [{filename}] at line number: [{line_number}] with error message: [{str(error)}]"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_details)

    def __str__(self):
        return self.error_message
    

def main():
    try:
        a = 1/0
    except Exception as e:
        raise CustomException(e, sys)