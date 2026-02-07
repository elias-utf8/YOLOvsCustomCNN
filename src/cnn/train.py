# train.py
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import configparser
from pathlib import Path
from datetime import datetime
import json
from model import MultiObjectDetector
from losses import DetectionLoss
from dataset import get_loaders

# Config
PROJECT_ROOT = Path(__file__).parents[2]
CONFIG = configparser.ConfigParser()
CONFIG.read(PROJECT_ROOT / 'config.ini')
DATASET_DIR = PROJECT_ROOT / CONFIG.get('data', 'DATASET_DIR')
MODELS_DIR = PROJECT_ROOT / CONFIG.get('cnn', 'MODELS_DIR')
RUNS_DIR = PROJECT_ROOT / CONFIG.get('cnn', 'RUNS_DIR')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16


def train_epoch(model, criterion, optimizer, loader):
    model.train()
    total_loss, total_cls, total_reg = 0, 0, 0
    
    for images, cls_target, reg_target in tqdm(loader, desc="Training"):
        images = images.to(device)
        cls_target = cls_target.to(device)
        reg_target = reg_target.to(device)
        
        # Forward
        cls_pred, reg_pred = model(images)
        loss, cls_loss, reg_loss = criterion(cls_pred, reg_pred, cls_target, reg_target)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        total_cls += cls_loss.item() * images.size(0)
        total_reg += reg_loss.item() * images.size(0)
    
    n = len(loader.dataset)
    return total_loss / n, total_cls / n, total_reg / n


@torch.no_grad()
def eval_epoch(model, criterion, loader):
    model.eval()
    total_loss, total_cls, total_reg = 0, 0, 0
    
    for images, cls_target, reg_target in loader:
        images = images.to(device)
        cls_target = cls_target.to(device)
        reg_target = reg_target.to(device)
        
        cls_pred, reg_pred = model(images)
        loss, cls_loss, reg_loss = criterion(cls_pred, reg_pred, cls_target, reg_target)
        
        total_loss += loss.item() * images.size(0)
        total_cls += cls_loss.item() * images.size(0)
        total_reg += reg_loss.item() * images.size(0)
    
    n = len(loader.dataset)
    return total_loss / n, total_cls / n, total_reg / n


def main(train_loader, val_loader, num_epochs=50, lr=0.001, patience=7):
    print(f"Device: {device}")

    # Créer les dossiers de sortie
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Modèle
    model = MultiObjectDetector(num_classes=2).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modele: {num_params:,} parametres")

    # Loss et optimizer
    criterion = DetectionLoss(cls_weight=1.0, reg_weight=20.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # Early stopping 
    best_val = float('inf')
    patience_counter = 0
    best_model_path = MODELS_DIR / 'best.pth'

    print(f"\nEntrainement (early stopping patience={patience})\n")

    history = {'train_loss': [], 'val_loss': [], 'train_cls': [],
               'val_cls': [], 'train_reg': [], 'val_reg': []}

    for epoch in range(num_epochs):
        train_loss, train_cls, train_reg = train_epoch(model, criterion, optimizer, train_loader)
        val_loss, val_cls, val_reg = eval_epoch(model, criterion, val_loader)

        # Enregistrer
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_cls'].append(train_cls)
        history['val_cls'].append(val_cls)
        history['train_reg'].append(train_reg)
        history['val_reg'].append(val_reg)

        # Scheduler
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:02d}/{num_epochs} | "
              f"Train: {train_loss:.4f} (cls:{train_cls:.4f}, reg:{train_reg:.4f}) | "
              f"Val: {val_loss:.4f}")

        # Early stopping logic
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  Meilleur modele sauvegarde: {best_model_path}")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"\nEarly stopping a l'epoch {epoch+1}")
                break

    print(f"\nTermine! Meilleure val_loss: {best_val:.4f}")

    # Sauvegarder l'historique et les courbes dans runs/
    save_training_outputs(history, run_dir)

    return model, history


def save_training_outputs(history, run_dir):
    """Sauvegarde l'historique et les courbes dans le dossier run."""
    # Sauvegarder l'historique en JSON
    history_path = run_dir / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Historique sauvegarde: {history_path}")

    # Générer et sauvegarder les courbes
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('Loss Totale')
    axes[0].legend()

    axes[1].plot(history['train_cls'], label='Train')
    axes[1].plot(history['val_cls'], label='Val')
    axes[1].set_title('Loss Classification')
    axes[1].legend()

    axes[2].plot(history['train_reg'], label='Train')
    axes[2].plot(history['val_reg'], label='Val')
    axes[2].set_title('Loss Regression')
    axes[2].legend()

    plt.tight_layout()
    curves_path = run_dir / 'training_curves.png'
    plt.savefig(curves_path)
    print(f"Courbes sauvegardees: {curves_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch",  type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_loaders(
        data_dir=str(DATASET_DIR),
        batch_size=args.batch
    )

    main(train_loader, val_loader, num_epochs=args.epochs)