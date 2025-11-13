"""
Utility functions for sentiment classification project
"""
import torch
import numpy as np
import random
import psutil
import platform

def set_seeds(seed=42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_system_info():
    """
    Get system hardware information
    
    Returns:
        Dictionary with system information
    """
    info = {
        'platform': platform.system(),
        'platform_version': platform.version(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'ram_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'cpu_count': psutil.cpu_count(logical=False),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'cuda_available': torch.cuda.is_available()
    }
    
    if info['cuda_available']:
        info['cuda_version'] = torch.version.cuda
        info['gpu_name'] = torch.cuda.get_device_name(0)
    
    return info

def print_system_info():
    """Print system information"""
    info = get_system_info()
    
    print("\n" + "="*70)
    print("SYSTEM INFORMATION")
    print("="*70)
    print(f"Platform: {info['platform']}")
    print(f"Processor: {info['processor']}")
    print(f"Python Version: {info['python_version']}")
    print(f"RAM: {info['ram_gb']} GB")
    print(f"CPU Cores: {info['cpu_count']} physical, {info['cpu_count_logical']} logical")
    print(f"CUDA Available: {info['cuda_available']}")
    if info['cuda_available']:
        print(f"CUDA Version: {info['cuda_version']}")
        print(f"GPU: {info['gpu_name']}")
    print("="*70 + "\n")

def save_system_info(filepath='./results/system_info.txt'):
    """Save system information to file"""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    info = get_system_info()
    
    with open(filepath, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SYSTEM INFORMATION\n")
        f.write("="*70 + "\n")
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
        f.write("="*70 + "\n")
    
    print(f"✓ System info saved to {filepath}")

def format_time(seconds):
    """
    Format seconds into human-readable time
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"

def calculate_model_size(model):
    """
    Calculate model size in MB
    
    Args:
        model: PyTorch model
    
    Returns:
        Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

def binary_accuracy(preds, y):
    """
    Calculate binary accuracy
    
    Args:
        preds: Predictions (logits)
        y: True labels
    
    Returns:
        Accuracy as float
    """
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

def count_parameters(model):
    """
    Count trainable parameters in model
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model):
    """
    Print model summary
    
    Args:
        model: PyTorch model
    """
    print("\n" + "="*70)
    print("MODEL SUMMARY")
    print("="*70)
    print(model)
    print("-"*70)
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"Model size: {calculate_model_size(model):.2f} MB")
    print("="*70 + "\n")

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=3, min_delta=0):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

if __name__ == '__main__':
    # Test utilities
    print_system_info()
    save_system_info()