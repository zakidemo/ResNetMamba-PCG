import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report

from models.wavemamba_1d import WaveMamba1D_Classifier
from utils.dataset import PhysioNet2016Dataset

def evaluate_model():
    # 1. Setup
    BASE_DIR = "data/classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0"
    FOLDERS =['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Re-create the exact same validation split
    full_dataset = PhysioNet2016Dataset(base_dir=BASE_DIR, folders=FOLDERS)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # NOTE: To ensure we test on the exact same split, we use a fixed manual seed
    generator = torch.Generator().manual_seed(42) 
    _, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 3. Load the Trained Model
    print("\nLoading Best WaveMamba Model...")
    model = WaveMamba1D_Classifier(sequence_length=16000, d_model=64, num_classes=2).to(device)
    model.load_state_dict(torch.load("models/best_wavemamba.pth"))
    model.eval()

    # 4. Run Inference
    all_preds = []
    all_labels =[]
    
    print("Testing on Validation Set...")
    with torch.no_grad():
        for signals, labels in val_loader:
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. Calculate IEEE Metrics
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # True Positive Rate (Abnormal)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # True Negative Rate (Normal)
    macc = (sensitivity + specificity) / 2                 # Official PhysioNet Score
    
    print("\n" + "="*50)
    print(" 🏥 OFFICIAL IEEE EVALUATION METRICS 🏥")
    print("="*50)
    print(f"Confusion Matrix:\n{cm}\n")
    print(f"Sensitivity (Detecting Abnormal) : {sensitivity*100:.2f}%")
    print(f"Specificity (Detecting Normal)   : {specificity*100:.2f}%")
    print(f"MAcc (Mean Accuracy - PhysioNet) : {macc*100:.2f}%")
    print("="*50)
    
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Normal (0)", "Abnormal (1)"]))

if __name__ == "__main__":
    evaluate_model()