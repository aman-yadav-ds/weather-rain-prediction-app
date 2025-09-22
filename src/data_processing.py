import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class DataProcessing:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        
        os.makedirs(self.output_path, exist_ok=True)
        logger.info("Data Processing Initialized...")

    def load_data(self):
        try:
            df = pd.read_csv(self.input_path)
            logger.info("Data Loaded Successfully.")
            return df
        except Exception as e:
            logger.error(f"Error while loading data {e}.")
            raise CustomException("Failed to load data.", e)
        
    def preprocess(self, df):
        try:
            categorical = []
            numerical = []

            for col in df.columns:
                if df[col].dtype == 'object':
                    categorical.append(col) 
                else:
                    numerical.append(col)

            df["Date"] = pd.to_datetime(df["Date"])

            df["Year"] = df["Date"].dt.year
            df["Month"] = df["Date"].dt.month
            df["Day"] = df["Date"].dt.day

            df.drop("Date" , axis=1 , inplace=True)

            for col in numerical:
                df[col].fillna(df[col].mean() , inplace=True)
            
            df.dropna(inplace=True)

            logger.info("Basic Data Processing Done.")
            return df
        except Exception as e:
            logger.error(f"Error while preprocessing data {e}.")
            raise CustomException("Failed to preprocess data.", e)
        
    def label_encode(self, df):
        try:
            categorical = [
                'Location',
                'WindGustDir',
                'WindDir9am',
                'WindDir3pm',
                'RainToday',
                'RainTomorrow']
            for col in categorical:
                label_encoder = LabelEncoder()
                df[col] = label_encoder.fit_transform(df[col])
                label_mapping = dict(zip(label_encoder.classes_ , range(len(label_encoder.classes_))))
                logger.info(f"Label Maping for {col}")
                logger.info(label_mapping)
            
            logger.info("Label Encoding Done.")
            return df
        except Exception as e:
            logger.error(f"Error while Label Encoding data {e}.")
            raise CustomException("Failed to Label Encode data.", e)
        
    def split_data(self, df):
        try:
            X = df.drop('RainTomorrow' , axis=1)
            Y = df["RainTomorrow"]

            logger.info(f"Cols: {X.columns}")
            X_train , X_test , y_train , y_test = train_test_split(X,Y , test_size=0.2 , random_state=42)

            joblib.dump(X_train, os.path.join(self.output_path, "X_train.pkl"))
            joblib.dump(X_test, os.path.join(self.output_path, "X_test.pkl"))
            joblib.dump(y_train, os.path.join(self.output_path, "y_train.pkl"))
            joblib.dump(y_test, os.path.join(self.output_path, "y_test.pkl"))

            logger.info("Splitted and saved sucessfully.")

        except Exception as e:
            logger.error(f"Error while Spliting data {e}.")
            raise CustomException("Failed to split data.", e)
        
    def run(self):
        df = self.load_data()
        df = self.preprocess(df)
        df = self.label_encode(df)
        self.split_data(df)

        logger.info("Data Processing Completed.")


if __name__ == "__main__":
    processor = DataProcessing(input_path="artifacts/raw/data.csv", output_path="artifacts/processed")
    processor.run()