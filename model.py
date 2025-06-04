import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import optuna
from optuna.trial import TrialState
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Data Loading and Preparation
def load_and_prepare_data(filepath):
    """Load and preprocess the dataset"""
    data = pd.read_csv(filepath)
    
    # Convert stringified lists to actual Python lists
    data['genres'] = data['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Extract the first genre as our label
    data['genres'] = data['genres'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
    
    # Combine title and summary for text input
    texts = data['title'] + " " + data['summary'].fillna('')
    labels = data['genres'].fillna('Unknown')
    
    return texts, labels

# 2. Dataset Class
class MovieGenreDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx] if isinstance(self.texts, pd.Series) else self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 3. Optuna Optimization Function
def objective(trial, texts, labels, tokenizer):
    """Optuna objective function for hyperparameter optimization"""
    # Print trial information
    print(f"\nStarting trial {trial.number}...")
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8])
    num_epochs = trial.suggest_int("num_epochs", 1, 1)
    max_len = trial.suggest_categorical("max_len", [128])
    
    # learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    # batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    # num_epochs = trial.suggest_int("num_epochs", 2, 5)
    # max_len = trial.suggest_categorical("max_len", [128, 256, 512])
    
    print(f"Trial {trial.number} parameters:")
    print(f"  learning_rate: {learning_rate:.2e}")
    print(f"  batch_size: {batch_size}")
    print(f"  num_epochs: {num_epochs}")
    print(f"  max_len: {max_len}")
    
    # Split data (do this inside objective to get different splits per trial)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Initialize model
    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels=len(label_encoder.classes_)
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Create data loaders
    train_dataset = MovieGenreDataset(X_train, y_train_encoded, tokenizer, max_len)
    test_dataset = MovieGenreDataset(X_test, y_test_encoded, tokenizer, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Training loop with progress indication
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Trial {trial.number} Epoch {epoch + 1}/{num_epochs}', leave=False)
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_loss / len(train_loader)
        print(f'Trial {trial.number} Epoch {epoch + 1} - Training loss: {avg_train_loss:.4f}')
    
    # Validation
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f'Trial {trial.number} Validating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Trial {trial.number} completed with accuracy: {accuracy:.4f}")
    return accuracy

# 4. Main Training Function
def train_model(texts, labels, best_params=None):
    """Train the model with optional best parameters from Optuna"""
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels_encoded, test_size=0.2, random_state=42
    )
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # If no best_params provided, use defaults
    if best_params is None:
        best_params = {
            'learning_rate': 2e-5,
            'batch_size': 16,
            'num_epochs': 3,
            'max_len': 256
        }
    
    # Initialize model
    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels=len(label_encoder.classes_)
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=best_params['learning_rate'])
    
    # Create data loaders
    train_dataset = MovieGenreDataset(X_train, y_train, tokenizer, best_params['max_len'])
    test_dataset = MovieGenreDataset(X_test, y_test, tokenizer, best_params['max_len'])
    
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'])
    
    # Training loop
    for epoch in range(best_params['num_epochs']):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{best_params["num_epochs"]}', leave=False)
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1} - Training loss: {avg_train_loss:.4f}')
        
        # Validation
        model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                _, preds = torch.max(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        present_labels = np.unique(true_labels)
        present_classes = label_encoder.classes_[present_labels]

        print(classification_report(
            true_labels,
            predictions,
            labels=present_labels,  # Only evaluate on present classes
            target_names=present_classes,  # Only show names for present classes
            zero_division=0
        ))
    
    return model, tokenizer, label_encoder

# 5. Prediction Function
def predict_genre(text, model, tokenizer, label_encoder, max_len=256):
    """Predict genre for a single text input"""
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    _, prediction = torch.max(outputs.logits, dim=1)
    return label_encoder.inverse_transform(prediction.cpu().numpy())[0]

# 6. Save and Load Functions
def save_model(model, tokenizer, label_encoder, save_dir):
    """Save model, tokenizer and label encoder"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save model and tokenizer
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # Save label encoder
    np.save(os.path.join(save_dir, 'label_encoder_classes.npy'), label_encoder.classes_)
    np.save(os.path.join(save_dir, 'label_encoder_params.npy'), label_encoder.get_params())

def load_model(save_dir):
    """Load model, tokenizer and label encoder"""
    # Load model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(save_dir)
    model = RobertaForSequenceClassification.from_pretrained(save_dir).to(device)
    
    # Load label encoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(os.path.join(save_dir, 'label_encoder_classes.npy'), allow_pickle=True)
    label_encoder.set_params(**np.load(os.path.join(save_dir, 'label_encoder_params.npy'), allow_pickle=True).item())
    
    return model, tokenizer, label_encoder

# Main Execution
if __name__ == "__main__":
    # Load and prepare data
    texts, labels = load_and_prepare_data("media_data.csv")
    
    # Initialize tokenizer for Optuna
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Run Optuna optimization (comment out if you just want to train with defaults)
    print("Running Optuna optimization...")
    
    # Add a callback function to print progress
    def print_trial_status(study, trial):
        print(f"\nTrial {trial.number} completed with value: {trial.value:.4f}")
        print(f"Best trial until now: {study.best_trial.number} with value: {study.best_trial.value:.4f}")
        print(f"Current best parameters: {study.best_trial.params}")
    
    study = optuna.create_study(direction="maximize")
    print(f"Starting optimization with {1} trials...")
    study.optimize(
        lambda trial: objective(trial, texts, labels, tokenizer),
        n_trials=1,
        callbacks=[print_trial_status],
        gc_after_trial=True
    )
    
    print("\nOptimization completed!")
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Accuracy: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    best_params = trial.params
    model, tokenizer, label_encoder = train_model(texts, labels, best_params)
    
    # Save the model
    save_dir = "./movie_genre_classifier"
    save_model(model, tokenizer, label_encoder, save_dir)
    print(f"Model saved to {save_dir}")
    
    # Example prediction
    sample_text = """A young wizard named Harry Potter discovers his magical heritage and begins 
                   his education at Hogwarts School of Witchcraft and Wizardry, where he makes 
                   friends, battles dark forces, and uncovers secrets about his past."""
    
    # Load the saved model (demonstration)
    loaded_model, loaded_tokenizer, loaded_label_encoder = load_model(save_dir)
    
    predicted_genre = predict_genre(
        sample_text, 
        loaded_model, 
        loaded_tokenizer, 
        loaded_label_encoder
    )
    print(f"\nPredicted genre for sample text: {predicted_genre}")