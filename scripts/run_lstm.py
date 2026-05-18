import os
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class SaccadeDataset(Dataset):
    def __init__(self, data_list, window_size=30):
        self.samples = []
        for seq in data_list:
            if len(seq) > window_size:
                for i in range(len(seq) - window_size):
                    x = seq[i : i + window_size]
                    y = seq[i + window_size]
                    self.samples.append((x, y))
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return (
            torch.tensor(x, dtype=torch.float32).unsqueeze(-1), 
            torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        )

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        out, _ = self.lstm(x)
        # out shape: [batch_size, seq_len, hidden_size]
        # We take the output of the last time step
        out = out[:, -1, :]
        return self.fc(out)

def main():
    saccades_path = os.path.join("data", "processed_saccades.csv")
    if not os.path.exists(saccades_path):
        print(f"Error: {saccades_path} does not exist. Run preprocess.py first.")
        return
        
    print("Loading processed saccades...")
    df = pd.read_csv(saccades_path)
    
    # Sort and group
    df_sorted = df.sort_values(by=['subject', 'round', 'session', 'onset_time'])
    grouped = df_sorted.groupby(['subject', 'round', 'task'])
    
    subjects = sorted(list(df['subject'].unique()))
    print(f"Total unique subjects found: {len(subjects)}")
    
    # Subject-disjoint split: 40 train/val subjects, 10 held-out test subjects
    train_val_subjects = subjects[:40]
    test_subjects = subjects[40:]
    
    # Further split train_val into train and validation (35 train, 5 validation)
    train_subjects = train_val_subjects[:35]
    val_subjects = train_val_subjects[35:]
    
    print(f"Disjoint Split: {len(train_subjects)} Train, {len(val_subjects)} Val, {len(test_subjects)} Test subjects")
    print(f"Held-out test subjects: {test_subjects}")
    
    train_seqs = []
    val_seqs = []
    
    # Scale factor: saccadic peak velocities are in range 50-950, so divide by 1000.0
    SCALE = 1000.0
    
    for (subj, rnd, tsk), group in grouped:
        vel_seq = group['peak_velocity'].values / SCALE
        if len(vel_seq) < 50:
            continue
            
        n_train = int(len(vel_seq) * 0.8)
        train_portion = vel_seq[:n_train]
        
        if subj in train_subjects:
            train_seqs.append(train_portion)
        elif subj in val_subjects:
            val_seqs.append(train_portion)
            
    print(f"Collected {len(train_seqs)} training sequences, {len(val_seqs)} validation sequences.")
    
    # Create datasets and dataloaders
    window_size = 30
    train_dataset = SaccadeDataset(train_seqs, window_size=window_size)
    val_dataset = SaccadeDataset(val_seqs, window_size=window_size)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = LSTMModel(input_size=1, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training Loop
    epochs = 50
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_state = None
    
    print("Training PyTorch LSTM model...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
            
        train_loss /= len(train_dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                predictions = model(x_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item() * x_batch.size(0)
        val_loss /= len(val_dataset)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
                
    # Load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    # Evaluate autoregressively on held-out test subjects
    model.eval()
    test_metrics = []
    
    print("Evaluating LSTM autoregressively on held-out test subjects...")
    for (subj, rnd, tsk), group in grouped:
        if subj not in test_subjects:
            continue
            
        vel_seq = group['peak_velocity'].values
        if len(vel_seq) < 50:
            continue
            
        # Split: Train 80%, Test 20%
        n_train = int(len(vel_seq) * 0.8)
        train_portion = vel_seq[:n_train]
        test_portion = vel_seq[n_train:]
        n_test = len(test_portion)
        
        # We need at least window_size values in train portion to prompt the LSTM
        if len(train_portion) < window_size:
            continue
            
        # Start rolling forecast
        history = list(train_portion[-window_size:] / SCALE)
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_test):
                x_input = torch.tensor(history[-window_size:], dtype=torch.float32).view(1, window_size, 1).to(device)
                pred_val = model(x_input).item()
                predictions.append(pred_val * SCALE)
                history.append(pred_val)
                
        predictions = np.array(predictions)
        
        # Calculate MAE and RMSE
        mae = float(np.mean(np.abs(test_portion - predictions)))
        rmse = float(np.sqrt(np.mean((test_portion - predictions)**2)))
        
        test_metrics.append({
            'subject': subj,
            'round': rnd,
            'task': tsk,
            'n_train': n_train,
            'n_test': n_test,
            'MAE': mae,
            'RMSE': rmse
        })
        
    # Summarize results
    maes = [m['MAE'] for m in test_metrics]
    rmses = [m['RMSE'] for m in test_metrics]
    
    mean_mae = float(np.mean(maes))
    mean_rmse = float(np.mean(rmses))
    std_mae = float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0
    std_rmse = float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0
    
    ci_mae = float(1.96 * std_mae / np.sqrt(len(maes))) if len(maes) > 1 else 0.0
    ci_rmse = float(1.96 * std_rmse / np.sqrt(len(rmses))) if len(rmses) > 1 else 0.0
    
    results = {
        'mean_MAE': mean_mae,
        'mean_RMSE': mean_rmse,
        'ci_MAE': ci_mae,
        'ci_rmse': ci_rmse,
        'n_series_evaluated': len(test_metrics),
        'series_results': test_metrics
    }
    
    lstm_results_path = os.path.join("data", "lstm_results.json")
    print(f"Saving LSTM results to {lstm_results_path}...")
    with open(lstm_results_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"LSTM evaluation complete: Mean MAE = {mean_mae:.4f}, Mean RMSE = {mean_rmse:.4f}")

if __name__ == "__main__":
    main()
