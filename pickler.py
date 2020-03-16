import joblib

class Pickle:

    def __init__(self):
        super().__init__()

    @staticmethod
    def pickle_model(my_model, name):
        joblib.dump(my_model, name+".pkl")

    @staticmethod
    def load_model(my_model_name):
        my_model_loaded = joblib.load(my_model_name+".pkl")
        return my_model_loaded
