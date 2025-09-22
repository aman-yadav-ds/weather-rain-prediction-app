import os
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.model = xgb.XGBClassifier()
        
        os.makedirs(self.output_path, exist_ok=True)

        logger.info("Model Training Initialized.")

    def load_data(self):
        try:
            X_train = joblib.load(os.path.join(self.input_path, "X_train.pkl"))
            X_test = joblib.load(os.path.join(self.input_path, "X_test.pkl"))
            y_train = joblib.load(os.path.join(self.input_path, "y_train.pkl"))
            y_test = joblib.load(os.path.join(self.input_path, "y_test.pkl"))

            logger.info("Data loaded successfully.")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error while loading Data {e}")
            raise CustomException("Failed to load data", e)

    def train_model(self, X_train, y_train):
        try:
            self.model.fit(X_train, y_train)
            joblib.dump(self.model, os.path.join(self.output_path, "model.pkl"))

            logger.info(f"Training and saving model done.")
        
        except Exception as e:
            logger.error(f"Error while Training Data {e}")
            raise CustomException("Failed to Train data", e)
        
    def evaluate_model(self,X_train, X_test, y_train, y_test):
        try:
            training_score = self.model.score(X_train, y_train)
            logger.info(f"Training Model Score is: {training_score}")

            y_pred = self.model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")

            logger.info(f"Accuracy: {accuracy} | Precision: {precision} | Recall: {recall} | F1 Score: {f1}")

            logger.info("Model Evaluation Done.")
        except Exception as e:
            logger.error(f"Error while Training Data {e}")
            raise CustomException("Failed to Train data", e)

    def run(self):
        X_train, X_test, y_train, y_test = self.load_data()
        self.train_model(X_train, y_train)
        self.evaluate_model(X_train, X_test, y_train, y_test)

        logger.info("Model training and Evaluation done.")

if __name__ == "__main__":
    trainer = ModelTraining(input_path="artifacts/processed", output_path="artifacts/models")

    trainer.run()