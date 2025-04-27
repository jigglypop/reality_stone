import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# GPU information and checks
print("\n===== GPU/CUDA Information =====")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Number of GPUs: {device_count}")
    for i in range(device_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Current memory usage: {torch.cuda.memory_allocated(i)/1024**2:.2f} MB")
        print(f"  Maximum memory capacity: {torch.cuda.get_device_properties(i).total_memory/1024**2:.2f} MB")
else:
    print("CUDA is not available. GPU not found or CUDA not properly installed.")
print("==============================\n")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# GPU optimization settings
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()
    print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB in use")

# Hyperbolic operations class
class HyperbolicOperations:
    @staticmethod
    def euclidean_to_poincare(x, c=1.0, max_norm=0.9):
        """
        Safely convert Euclidean vectors to Poincare ball
        x: batch of Euclidean vectors [batch_size, dim]
        c: curvature
        max_norm: maximum norm (to prevent points from getting too close to the boundary)
        """
        # Calculate norm [batch_size, 1]
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        
        # Handle zero norms
        zeros_mask = (norm == 0)
        safe_norm = torch.where(zeros_mask, torch.ones_like(norm), norm)
        
        # Calculate scale considering curvature [batch_size, 1]
        sqrt_c = torch.sqrt(torch.tensor(c, device=x.device, dtype=x.dtype))
        scale = max_norm * torch.tanh(sqrt_c * norm) / (sqrt_c * safe_norm)
        
        # Scale vector, returning zero vector for zero norm
        return torch.where(zeros_mask, torch.zeros_like(x), scale * x)
    
    @staticmethod
    def poincare_to_euclidean(x, c=1.0):
        """
        Convert Poincare ball vectors to Euclidean space (approximate log map from origin)
        x: batch of Poincare vectors [batch_size, dim]
        c: curvature
        """
        # Calculate and clamp norms [batch_size, 1]
        norms = torch.norm(x, p=2, dim=-1, keepdim=True)
        norms = torch.clamp(norms, min=1e-8, max=1.0-1e-8)
        
        # Convert to tensor
        sqrt_c = torch.sqrt(torch.tensor(c, device=x.device, dtype=x.dtype))
        
        # Convert to Euclidean space [batch_size, dim]
        return x * torch.atanh(sqrt_c * norms) / (sqrt_c * norms)
    
    @staticmethod
    def batch_poincare_exp_map(x, v, c=1.0):
        """
        Batch exponential map in Poincare ball
        x: batch reference points [batch_size, dim]
        v: batch tangent vectors [batch_size, dim]
        c: curvature
        """
        eps = 1e-8
        
        # Norm squared of reference points [batch_size, 1]
        x_norm_squared = torch.sum(x * x, dim=-1, keepdim=True)
        
        # Conformal factor [batch_size, 1]
        lambda_x = 2.0 / (1.0 - c * x_norm_squared + eps)
        
        # Norm of tangent vectors [batch_size, 1]
        v_norm = torch.norm(v, p=2, dim=-1, keepdim=True)
        v_norm = torch.clamp(v_norm, min=eps)
        
        # Convert c to tensor
        c_tensor = torch.tensor(c, device=x.device, dtype=x.dtype)
        sqrt_c = torch.sqrt(c_tensor)
        
        # Calculate scale factor [batch_size, 1]
        scale = torch.tanh(sqrt_c * lambda_x * v_norm / 2.0) / (sqrt_c * v_norm)
        
        # Scale vectors [batch_size, dim]
        scaled_v = scale * v
        
        # Calculate numerator [batch_size, dim]
        numerator = (1.0 - c * x_norm_squared) * scaled_v
        
        # Calculate denominator [batch_size, 1]
        x_scaled_v_inner = torch.sum(x * scaled_v, dim=-1, keepdim=True)
        scaled_v_norm_squared = torch.sum(scaled_v * scaled_v, dim=-1, keepdim=True)
        denominator = 1.0 - 2.0 * c * x_scaled_v_inner + c * c * x_norm_squared * scaled_v_norm_squared
        
        # Result (Mobius addition) [batch_size, dim]
        result = x + numerator / (denominator + eps)
        
        # Check numerical stability
        mask = torch.isfinite(result).all(dim=-1, keepdim=True)
        result = torch.where(mask, result, torch.zeros_like(result))
        
        return result
    
    @staticmethod
    def batch_poincare_log_map(x, y, c=1.0):
        """
        Batch logarithmic map in Poincare ball
        x: batch reference points [batch_size, dim]
        y: batch target points [batch_size, dim]
        c: curvature
        """
        eps = 1e-8
        
        # Norm squared of reference points [batch_size, 1]
        x_norm_squared = torch.sum(x * x, dim=-1, keepdim=True)
        
        # Conformal factor [batch_size, 1]
        lambda_x = 2.0 / (1.0 - c * x_norm_squared + eps)
        
        # For Mobius subtraction [batch_size, 1]
        y_norm_squared = torch.sum(y * y, dim=-1, keepdim=True)
        xy_inner_prod = torch.sum(x * y, dim=-1, keepdim=True)
        
        # Calculate numerator [batch_size, dim]
        numerator = (1.0 - 2.0 * c * xy_inner_prod + c * y_norm_squared) * x
        numerator = numerator - (1.0 - c * x_norm_squared) * y
        
        # Calculate denominator [batch_size, 1]
        denominator = 1.0 - 2.0 * c * xy_inner_prod + c * c * x_norm_squared * y_norm_squared
        
        # Difference vector [batch_size, dim]
        diff = numerator / (denominator + eps)
        
        # Norm of difference vector [batch_size, 1]
        diff_norm = torch.norm(diff, p=2, dim=-1, keepdim=True)
        diff_norm = torch.clamp(diff_norm, min=eps)
        
        # Convert c to tensor
        c_tensor = torch.tensor(c, device=x.device, dtype=x.dtype)
        sqrt_c = torch.sqrt(c_tensor)
        
        # Final result [batch_size, dim]
        return 2.0 / (sqrt_c * lambda_x) * torch.atanh(sqrt_c * diff_norm) * diff / diff_norm

# Load MNIST dataset
def load_mnist(batch_size=128, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # DataLoader with num_workers and pin_memory for better GPU performance
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, test_loader

# Standard MLP model (Euclidean space)
class EuclideanMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super(EuclideanMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Hyperbolic MLP model
class HyperbolicMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10, curvature=0.1):
        super(HyperbolicMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.curvature = curvature
        
        # Input layer: Euclidean to intermediate
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Hidden layer (in hyperbolic space)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Output layer
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Hyperbolic operations
        self.hyper_ops = HyperbolicOperations()
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    @torch.cuda.amp.autocast(enabled=True)  # Mixed precision
    def forward(self, x):
        batch_size = x.size(0)
        
        # Flatten and first layer (Euclidean)
        x = x.view(batch_size, -1)
        h1 = F.relu(self.bn1(self.fc1(x)))
        
        # Convert to hyperbolic space
        h1_hyp = self.hyper_ops.euclidean_to_poincare(h1, self.curvature)
        
        # Create origin tensor for log/exp maps
        origin = torch.zeros_like(h1_hyp)
        
        # Map to tangent space at origin
        h1_tan = self.hyper_ops.batch_poincare_log_map(origin, h1_hyp, self.curvature)
        
        # Apply linear transformation in tangent space
        h2_tan = self.fc2(h1_tan)
        h2_tan = F.relu(self.bn2(h2_tan))
        
        # Map back to hyperbolic space
        h2_hyp = self.hyper_ops.batch_poincare_exp_map(origin, h2_tan, self.curvature)
        
        # Convert back to Euclidean for output layer
        h2_euc = self.hyper_ops.poincare_to_euclidean(h2_hyp, self.curvature)
        
        # Output layer
        out = self.fc3(h2_euc)
        
        return F.log_softmax(out, dim=1)

# Mixed Precision Trainer
class MixedPrecisionTrainer:
    def __init__(self, model, optimizer, scaler=None):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler if scaler is not None else torch.cuda.amp.GradScaler()
    
    def train_step(self, data, target):
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            output = self.model(data)
            loss = F.nll_loss(output, target)
        
        # Scale gradients and backward pass
        self.scaler.scale(loss).backward()
        
        # Unscale for gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update with scaled gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return output, loss

# Train function
def train(model, train_loader, optimizer, epoch, device, log_interval=100, mixed_precision=True):
    model.train()
    start_time = time.time()
    
    # Training statistics
    total_loss = 0
    correct = 0
    total = 0
    
    # Mixed precision setup
    if mixed_precision and torch.cuda.is_available():
        trainer = MixedPrecisionTrainer(model, optimizer)
        print("Mixed precision training enabled")
    else:
        trainer = None
        print("Using standard precision training")
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        try:
            if trainer:
                # Use mixed precision
                output, loss = trainer.train_step(data, target)
            else:
                # Standard training
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)
            
            if batch_idx % log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                      f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                
                # Show GPU memory usage
                if device.type == 'cuda':
                    print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB / "
                          f"{torch.cuda.memory_reserved()/1024**2:.1f}MB (allocated/reserved)")
                
                # Check for NaN
                if torch.isnan(loss):
                    print("Warning: NaN loss detected!")
                    for name, param in model.named_parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            print(f"NaN gradient found in: {name}")
        
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    train_time = time.time() - start_time
    train_acc = 100. * correct / total if total > 0 else 0
    
    print(f'Train Epoch: {epoch} complete | Avg loss: {total_loss / (batch_idx + 1):.4f} | Accuracy: {train_acc:.2f}%')
    print(f'Training time: {train_time:.2f}s ({train_time/(batch_idx+1)*1000:.2f}ms per batch)')
    
    return train_acc, train_time

# Test function
def test(model, test_loader, device, mixed_precision=True):
    model.eval()
    test_loss = 0
    correct = 0
    start_time = time.time()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            try:
                # Mixed precision inference (optional)
                if mixed_precision and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        output = model(data)
                else:
                    output = model(data)
                
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            except Exception as e:
                print(f"Error during testing: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    test_time = time.time() - start_time
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    print(f'Testing time: {test_time:.2f}s\n')
    
    return accuracy, test_time

# Run experiment
def run_experiment(model_name, epochs=5, batch_size=128, mixed_precision=True):
    # Load data
    num_workers = min(4, os.cpu_count() or 1)
    train_loader, test_loader = load_mnist(batch_size=batch_size, num_workers=num_workers)
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()
        print(f"CUDA initial memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
    else:
        device = torch.device("cpu")
        print("Warning: Running on CPU")
        mixed_precision = False
    
    # Initialize model
    if model_name == 'euclidean':
        model = EuclideanMLP().to(device)
        print("Euclidean MLP model created")
    elif model_name == 'hyperbolic':
        model = HyperbolicMLP(curvature=0.1).to(device)
        print("Hyperbolic MLP model created")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Check model device
    print(f"Model device: {next(model.parameters()).device}")
    
    # Print model summary
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name}")
    print(f"Parameters: {num_params:,}")
    
    # Optimizer setup
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=0.001,
        weight_decay=1e-4,
        betas=(0.9, 0.99),
        amsgrad=True
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.5, 
        patience=1, 
        verbose=True,
        min_lr=1e-6
    )
    
    # Training and evaluation
    train_accs = []
    train_times = []
    test_accs = []
    test_times = []
    
    for epoch in range(1, epochs + 1):
        # Clear GPU memory before training
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        print(f"\n===== Epoch {epoch}/{epochs} =====")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_acc, train_time = train(model, train_loader, optimizer, epoch, device, 
                                      mixed_precision=mixed_precision)
        
        # Evaluate
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        test_acc, test_time = test(model, test_loader, device, mixed_precision=mixed_precision)
        
        # Adjust learning rate
        scheduler.step(test_acc)
        
        # Save results
        train_accs.append(train_acc)
        train_times.append(train_time)
        test_accs.append(test_acc)
        test_times.append(test_time)
        
        # Check GPU memory usage
        if device.type == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            print(f"Epoch {epoch} peak GPU memory usage: {peak_memory:.2f} MB")
            torch.cuda.reset_peak_memory_stats()
    
    # Print results
    print(f"\n--- {model_name} Final Results ---")
    print(f"Average training time (per epoch): {np.mean(train_times):.2f} seconds")
    print(f"Average inference time (test set): {np.mean(test_times):.2f} seconds")
    print(f"Final accuracy: {test_accs[-1]:.2f}%")
    
    return {
        'model': model_name,
        'train_accs': train_accs,
        'train_times': train_times,
        'test_accs': test_accs,
        'test_times': test_times,
        'num_params': num_params,
        'device': str(device)
    }

# Main function
if __name__ == "__main__":
    print("\n======= CUDA/CUDNN Settings =======")
    print(f"cudnn available: {torch.backends.cudnn.is_available()}")
    print(f"cudnn enabled: {torch.backends.cudnn.enabled}")
    
    # Check mixed precision support
    amp_supported = (
        torch.cuda.is_available() and 
        hasattr(torch.cuda, 'amp') and 
        hasattr(torch.cuda.amp, 'autocast')
    )
    print(f"Mixed precision supported: {amp_supported}")
    
    # Check tensor cores support
    tensor_cores_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
    print(f"Tensor cores supported: {tensor_cores_supported}")
    if tensor_cores_supported:
        print("Tensor core optimizations enabled (TF32/FP16)")
    
    # Experiment settings
    epochs = 5
    batch_size = 128
    mixed_precision = amp_supported
    
    print(f"Number of epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Mixed precision training: {mixed_precision}")
    print("===============================\n")
    
    # Run experiments
    print("\n[1/2] Starting Euclidean MLP experiment...")
    euclidean_results = run_experiment('euclidean', epochs=epochs, batch_size=batch_size, 
                                      mixed_precision=mixed_precision)
    
    print("\n[2/2] Starting Hyperbolic MLP experiment...")
    hyperbolic_results = run_experiment('hyperbolic', epochs=epochs, batch_size=batch_size, 
                                       mixed_precision=mixed_precision)
    
    # Visualize results
    epoch_range = range(1, epochs + 1)
    
    # Accuracy comparison graph
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(epoch_range, euclidean_results['test_accs'], 'bo-', label='Euclidean MLP')
    plt.plot(epoch_range, hyperbolic_results['test_accs'], 'ro-', label='Hyperbolic MLP')
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Training time comparison
    plt.subplot(2, 2, 2)
    models = ['Euclidean MLP', 'Hyperbolic\nMLP']
    train_times = [
        np.mean(euclidean_results['train_times']), 
        np.mean(hyperbolic_results['train_times'])
    ]
    plt.bar(models, train_times, color=['blue', 'red'])
    plt.title('Average Training Time (per epoch)')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    
    # Inference time comparison
    plt.subplot(2, 2, 3)
    test_times = [
        np.mean(euclidean_results['test_times']), 
        np.mean(hyperbolic_results['test_times'])
    ]
    plt.bar(models, test_times, color=['blue', 'red'])
    plt.title('Average Inference Time (test set)')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    
    # Parameter count comparison
    plt.subplot(2, 2, 4)
    params = [
        euclidean_results['num_params'],
        hyperbolic_results['num_params']
    ]
    plt.bar(models, params, color=['blue', 'red'])
    plt.title('Parameter Count')
    plt.ylabel('Count')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    
    # Learning curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epoch_range, euclidean_results['train_accs'], 'b--', label='Euclidean (Train)')
    plt.plot(epoch_range, euclidean_results['test_accs'], 'b-', label='Euclidean (Test)')
    plt.plot(epoch_range, hyperbolic_results['train_accs'], 'r--', label='Hyperbolic (Train)')
    plt.plot(epoch_range, hyperbolic_results['test_accs'], 'r-', label='Hyperbolic (Test)')
    plt.title('Train/Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('learning_curves.png')
    
    # Results summary
    print("\n=== Performance Comparison ===")
    print(f"{'Model':<25} {'Parameters':<12} {'Train Time':<15} {'Inference Time':<15} {'Accuracy':<10} {'Device':<10}")
    print("-" * 90)
    print(f"{'Euclidean MLP':<25} {euclidean_results['num_params']:,}<{'':<10} "
          f"{np.mean(euclidean_results['train_times']):.2f}s{'':<10} "
          f"{np.mean(euclidean_results['test_times']):.2f}s{'':<10} "
          f"{euclidean_results['test_accs'][-1]:.2f}%{'':<5} {euclidean_results['device']}")
    print(f"{'Hyperbolic MLP':<25} {hyperbolic_results['num_params']:,}<{'':<10} "
          f"{np.mean(hyperbolic_results['train_times']):.2f}s{'':<10} "
          f"{np.mean(hyperbolic_results['test_times']):.2f}s{'':<10} "
          f"{hyperbolic_results['test_accs'][-1]:.2f}%{'':<5} {hyperbolic_results['device']}")
    
    print("\nExperiment completed. Results saved to model_comparison.png and learning_curves.png") 