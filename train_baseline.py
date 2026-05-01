import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix

# Import ResNet instead of WaveMamba!
from models.resnet1d import ResNet1D
from utils.dataset import PhysioNet2016Dataset

def main():
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    EPOCHS = 30 # ResNet usually overfits quickly, 30 is enough
    BASE_DIR = "data/classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0"
    FOLDERS =['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training BASELINE RESNET on: {device.type.upper()}")

    # 1. Dataset (Same exact 80/20 split, using seed 42 so it's a fair comparison)
    print("Loading augmented dataset...")
    full_dataset = PhysioNet2016Dataset(base_dir=BASE_DIR, folders=FOLDERS, augment=True)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Crucial: manual_seed(42) ensures the ResNet takes the exact same exam as WaveMamba
    generator = torch.Generator().manual_seed(42) 
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 2. Model, Loss, Optimizer
    model = ResNet1D(num_classes=2).to(device)
    
    # Same Weighted Cross Entropy as WaveMamba to be completely fair
    class_weights = torch.tensor([1.0, 3.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 3. Training Loop
    best_macc = 0.0
    print("\n" + "="*50)
    print("BEGINNING RESNET BASELINE TRAINING...")
    print("="*50)
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # --- Train ---
        model.train()
        train_loss = 0.0
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # --- Evaluate ---
        model.eval()
        all_preds, all_labels = [],[]
        val_loss = 0.0
        
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        # Calculate IEEE Metrics for Baseline
        cm = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = cm.ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        macc = (sensitivity + specificity) / 2
        
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        time_taken = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Time: {time_taken:.1f}s | Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f}")
        print(f"   -> Sens: {sensitivity*100:.2f}% | Spec: {specificity*100:.2f}% | MAcc: {macc*100:.2f}%")
        
        if macc > best_macc:
            best_macc = macc
            torch.save(model.state_dict(), "models/best_resnet.pth")
            print("   🌟 New Best Baseline Model Saved!")
            
    print("\n🎉 BASELINE TRAINING COMPLETE!")
    print(f"Highest Baseline MAcc: {best_macc*100:.2f}%")

if __name__ == "__main__":
    main()