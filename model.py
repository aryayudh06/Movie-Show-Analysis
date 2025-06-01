import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from transformers import RobertaTokenizer, TFRobertaModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
import ast
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenrePredictorRoBERTa:
    def __init__(self, csv_path, max_length=128, batch_size=32, epochs=3):
        self.csv_path = csv_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.roberta_model = TFRobertaModel.from_pretrained('roberta-base')
        self.mlb = MultiLabelBinarizer()
        
    def load_and_preprocess_data(self):
        """
        Load data from CSV and preprocess for genre prediction
        """
        logger.info("Loading and preprocessing data...")
        
        # Load CSV data
        df = pd.read_csv(self.csv_path)
        
        # Clean and prepare data
        df = self._clean_data(df)
        
        # Extract features and labels
        texts = df['title'] + " " + df['summary'].fillna('')
        
        # Convert string representation of lists to actual lists
        genres = self._safe_convert_genres(df['genres'])
        
        # Multi-label binarization
        self.mlb.fit(genres)
        labels = self.mlb.transform(genres)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        logger.info(f"Data loaded. {len(df)} samples total.")
        logger.info(f"Number of unique genres: {len(self.mlb.classes_)}")
        logger.info(f"Train samples: {len(self.X_train)}, Test samples: {len(self.X_test)}")
        
    def _safe_convert_genres(self, genre_series):
        """
        Safely convert genre strings to lists, handling various formats
        """
        converted = []
        for item in genre_series:
            if pd.isna(item) or item == '[]' or item == '':
                converted.append([])
            elif isinstance(item, str):
                try:
                    # Handle string representation of list
                    parsed = ast.literal_eval(item)
                    if isinstance(parsed, list):
                        converted.append(parsed)
                    else:
                        # If it's a single string, make it a list
                        converted.append([parsed])
                except (ValueError, SyntaxError):
                    # If literal_eval fails, split by comma or treat as single genre
                    if ',' in item:
                        converted.append([g.strip() for g in item.split(',')])
                    else:
                        converted.append([item.strip()])
            elif isinstance(item, (list, np.ndarray)):
                converted.append(list(item))
            else:
                converted.append([])
        return converted
    
    def _clean_data(self, df):
        """
        Clean the input DataFrame
        """
        # Drop rows with missing critical data
        df = df.dropna(subset=['title'])
        
        # Fill missing summaries with empty string
        df['summary'] = df['summary'].fillna('')
        
        # Ensure genres column exists (create empty list if missing)
        if 'genres' not in df.columns:
            df['genres'] = [[] for _ in range(len(df))]
        
        return df
    
    def _encode_texts(self, texts):
        encoded = self.tokenizer(
            texts.tolist(),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='np'  # Changed from 'tf' to 'np'
        )
        return encoded
    
    def build_model(self):
        """
        Build the RoBERTa-based classification model
        """
        logger.info("Building model...")
        
        # Input layers
        input_ids = Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')
        
        # Create a custom layer to handle the RoBERTa model
        class RobertaLayer(tf.keras.layers.Layer):
            def __init__(self, roberta_model, **kwargs):
                super(RobertaLayer, self).__init__(**kwargs)
                self.roberta_model = roberta_model

            def call(self, inputs):
                input_ids, attention_mask = inputs
                outputs = self.roberta_model(input_ids, attention_mask=attention_mask)
                return outputs[1]  # Return the pooled output

            def get_config(self):
                config = super().get_config()
                # Note: We can't serialize the roberta_model itself, so we'll just save its config
                config.update({
                    "roberta_model_config": self.roberta_model.config.to_dict(),
                })
                return config

            @classmethod
            def from_config(cls, config):
                # Recreate the roberta_model from config
                roberta_config = config.pop("roberta_model_config")
                roberta_model = TFRobertaModel.from_pretrained('roberta-base', config=roberta_config)
                return cls(roberta_model, **config)
        
        # Use the custom layer
        roberta_layer = RobertaLayer(self.roberta_model)
        roberta_output = roberta_layer([input_ids, attention_mask])
        
        # Classification head
        x = Dropout(0.1)(roberta_output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.1)(x)
        outputs = Dense(len(self.mlb.classes_), activation='sigmoid')(x)
        
        # Compile model
        self.model = Model(inputs=[input_ids, attention_mask], outputs=outputs)
        self.model.compile(
            optimizer=Adam(learning_rate=3e-5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Model built successfully.")
        
    def train(self):
        """
        Train the model
        """
        logger.info("Preparing training data...")
        
        # Encode training data
        train_encoded = self._encode_texts(self.X_train)
        
        # Extract numpy arrays
        input_ids = train_encoded['input_ids']
        attention_mask = train_encoded['attention_mask']
        labels = self.y_train
        
        # Early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        
        logger.info("Training model...")
        self.history = self.model.fit(
            x={'input_ids': input_ids, 'attention_mask': attention_mask},
            y=labels,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.1,
            callbacks=[early_stopping],
            shuffle=True
        )
        
        logger.info("Training completed.")
    
    def evaluate(self):
        """
        Evaluate the model on test data
        """
        logger.info("Evaluating model...")
        
        # Encode test data
        test_encoded = self._encode_texts(self.X_test)
        
        # Extract numpy arrays from the BatchEncoding object
        input_ids = test_encoded['input_ids']
        attention_mask = test_encoded['attention_mask']
        
        # Predict using the extracted arrays
        y_pred = self.model.predict(
            {'input_ids': input_ids, 'attention_mask': attention_mask}
        )
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Print classification report
        report = classification_report(
            self.y_test,
            y_pred_binary,
            target_names=self.mlb.classes_
        )
        logger.info("\nClassification Report:\n" + report)
        
        return report
    
    def save_model(self, model_dir='genre_roberta_model.keras'):
        """
        Save the trained model
        """
        self.model.save(model_dir)
        logger.info(f"Model saved to {model_dir}")
    
    def predict_genres(self, text, threshold=0.5):
        """
        Predict genres for new text
        """
        # Encode input text
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='tf'
        )
        
        # Convert BatchEncoding to a format Keras can understand
        inputs = {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
        
        # Make prediction
        pred = self.model.predict(inputs)
        
        # Apply threshold and get genre names
        predicted_genres = [
            self.mlb.classes_[i] 
            for i, val in enumerate(pred[0]) 
            if val > threshold
        ]
        
        return predicted_genres

def main(csv_path='media_data.csv'):
    # Initialize genre predictor
    predictor = GenrePredictorRoBERTa(csv_path)
    
    try:
        # Load and preprocess data
        predictor.load_and_preprocess_data()
        
        # Build model
        predictor.build_model()
        
        # Train model
        predictor.train()
        
        # Evaluate model
        report = predictor.evaluate()
        
        # Save model
        predictor.save_model()
        
        # Example prediction
        example_text = "A group of astronauts travel through a wormhole in search of a new home for humanity"
        predicted_genres = predictor.predict_genres(example_text)
        logger.info(f"\nExample Prediction for: '{example_text}'")
        logger.info(f"Predicted genres: {predicted_genres}")
        
        return report
        
    except Exception as e:
        logger.error(f"Error in genre prediction pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the pipeline
    main()