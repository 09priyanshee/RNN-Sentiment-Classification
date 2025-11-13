"""
Data preprocessing for IMDb sentiment classification
"""
import re
import pickle
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

def set_seeds(seed=42):
    """Set random seeds for reproducibility"""
    import random
    import torch
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_imdb_data(num_words=10000):
    """
    Load IMDb dataset
    Returns raw reviews (as word indices) and labels
    """
    print(f"Loading IMDb dataset with top {num_words} words...")
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)
    
    # Get word index for reference
    word_index = imdb.get_word_index()
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    return (X_train, y_train), (X_test, y_test), word_index

def prepare_sequences(X_train, X_test, max_length=50):
    """
    Pad/truncate sequences to fixed length
    
    Args:
        X_train: Training sequences
        X_test: Test sequences
        max_length: Maximum sequence length
    
    Returns:
        Padded training and test sequences
    """
    print(f"Padding sequences to length {max_length}...")
    
    # Pad sequences
    X_train_pad = pad_sequences(X_train, maxlen=max_length, 
                                 padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test, maxlen=max_length, 
                                padding='post', truncating='post')
    
    print(f"Training shape: {X_train_pad.shape}")
    print(f"Test shape: {X_test_pad.shape}")
    
    return X_train_pad, X_test_pad

def get_dataset_statistics(X_train, X_test, y_train, y_test):
    """
    Calculate and print dataset statistics
    """
    # Original lengths before padding
    train_lengths = [len(x) for x in X_train]
    test_lengths = [len(x) for x in X_test]
    
    stats = {
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_avg_length': np.mean(train_lengths),
        'train_max_length': np.max(train_lengths),
        'train_min_length': np.min(train_lengths),
        'test_avg_length': np.mean(test_lengths),
        'positive_train': np.sum(y_train),
        'negative_train': len(y_train) - np.sum(y_train),
        'positive_test': np.sum(y_test),
        'negative_test': len(y_test) - np.sum(y_test)
    }
    
    print("\n=== Dataset Statistics ===")
    print(f"Training samples: {stats['train_samples']}")
    print(f"Test samples: {stats['test_samples']}")
    print(f"Average review length (train): {stats['train_avg_length']:.1f}")
    print(f"Max review length (train): {stats['train_max_length']}")
    print(f"Min review length (train): {stats['train_min_length']}")
    print(f"Positive/Negative (train): {stats['positive_train']}/{stats['negative_train']}")
    print(f"Positive/Negative (test): {stats['positive_test']}/{stats['negative_test']}")
    
    return stats

def preprocess_data(num_words=10000, max_length=50, save_dir='./data'):
    """
    Complete preprocessing pipeline
    
    Args:
        num_words: Vocabulary size
        max_length: Sequence length
        save_dir: Directory to save preprocessed data
    
    Returns:
        Preprocessed training and test data
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Set seeds
    set_seeds()
    
    # Load data
    (X_train, y_train), (X_test, y_test), word_index = load_imdb_data(num_words)
    
    # Get statistics before padding
    stats = get_dataset_statistics(X_train, X_test, y_train, y_test)
    
    # Prepare sequences
    X_train_pad, X_test_pad = prepare_sequences(X_train, X_test, max_length)
    
    # Save preprocessed data
    data_dict = {
        'X_train': X_train_pad,
        'y_train': y_train,
        'X_test': X_test_pad,
        'y_test': y_test,
        'word_index': word_index,
        'stats': stats,
        'max_length': max_length,
        'num_words': num_words
    }
    
    save_path = os.path.join(save_dir, f'preprocessed_len{max_length}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"\nPreprocessed data saved to {save_path}")
    
    return X_train_pad, y_train, X_test_pad, y_test

if __name__ == '__main__':
    # Preprocess for all sequence lengths
    for seq_len in [25, 50, 100]:
        print(f"\n{'='*60}")
        print(f"Preprocessing for sequence length: {seq_len}")
        print('='*60)
        preprocess_data(num_words=10000, max_length=seq_len)