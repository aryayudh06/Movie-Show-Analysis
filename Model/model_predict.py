import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

# Main execution for prediction
if __name__ == "__main__":
    # Load the saved model
    save_dir = "./movie_genre_classifier"  # Path to your saved model
    model, tokenizer, label_encoder = load_model(save_dir)
    print("Model loaded successfully!")
    
    sample_text = """A young wizard named Harry Potter discovers his magical heritage and begins 
                   his education at Hogwarts School of Witchcraft and Wizardry, where he makes 
                   friends, battles dark forces, and uncovers secrets about his past."""
    
    predicted_genre = predict_genre(
        sample_text, 
        model, 
        tokenizer, 
        label_encoder
    )
    print(f"\nPredicted genre: {predicted_genre}")