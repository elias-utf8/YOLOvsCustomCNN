# test.py
import torch
import numpy as np
from tqdm import tqdm
from model import MultiObjectDetector
from dataset import get_loaders
import matplotlib.pyplot as plt
import matplotlib.patches as patches

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
CLASSES = {0: 'Cube', 1: 'Cylindre'}
COLORS = {0: 'red', 1: 'blue'}



@torch.no_grad()
def evaluate_all_dataset(model, dataset, iou_threshold=0.5):
    """
    Évalue le modèle sur toutes les images du dataset.
    """
    model.eval()
    
    total_images = len(dataset)
    
    # Métriques par classe
    class_stats = {c: {'ious': [], 'tp': 0, 'fp': 0, 'fn': 0} for c in range(2)}
    
    # Métriques globales classification
    cls_correct = 0
    cls_total = 0
    
    print(f"Evaluation sur {total_images} images...\n")
    
    for i in tqdm(range(total_images), desc="Évaluation"):
        img, cls_target, reg_target = dataset[i]
        
        # Prédiction
        img_batch = img.unsqueeze(0).to(device)
        cls_pred, reg_pred = model(img_batch)
        
        # Convertir en binaire (présence/absence)
        cls_pred_binary = (torch.sigmoid(cls_pred[0]) > 0.5).cpu()
        reg_pred = reg_pred[0].cpu()
        
        # Évaluer chaque classe
        for c in range(2):
            pred_present = cls_pred_binary[c].item()
            true_present = cls_target[c].item()
            
            # Accuracy classification
            if pred_present == true_present:
                cls_correct += 1
            cls_total += 1
            
            # Détection metrics
            if true_present == 1 and pred_present == 1:
                # True Positive - calculer IoU
                iou = calculate_iou(reg_pred[c].numpy(), reg_target[c].numpy())
                class_stats[c]['ious'].append(iou)
                
                if iou >= iou_threshold:
                    class_stats[c]['tp'] += 1
                else:
                    # Détection mais mauvaise localisation
                    class_stats[c]['fp'] += 1
                    
            elif true_present == 0 and pred_present == 1:
                # False Positive
                class_stats[c]['fp'] += 1
                
            elif true_present == 1 and pred_present == 0:
                # False Negative
                class_stats[c]['fn'] += 1
    
    # Calculer les métriques finales
    all_ious = []
    for c in range(2):
        all_ious.extend(class_stats[c]['ious'])
    
    mean_iou = np.mean(all_ious) if all_ious else 0.0
    cls_accuracy = cls_correct / cls_total if cls_total > 0 else 0.0
    
    # Métriques par classe
    per_class = {}
    for c in range(2):
        tp = class_stats[c]['tp']
        fp = class_stats[c]['fp']
        fn = class_stats[c]['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        class_mean_iou = np.mean(class_stats[c]['ious']) if class_stats[c]['ious'] else 0.0
        
        per_class[c] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_iou': class_mean_iou,
            'tp': tp, 'fp': fp, 'fn': fn,
            'all_ious': class_stats[c]['ious']
        }
    
    results = {
        'mean_iou': mean_iou,
        'cls_accuracy': cls_accuracy,
        'total_images': total_images,
        'iou_threshold': iou_threshold,
        'all_ious': all_ious,
        'per_class': per_class
    }
    
    return results

def denormalize(img_tensor):
    """Remet l'image en format affichable."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


@torch.no_grad()
def plot_predictions(model, dataset, num_images=9, output_dir=None):
    """Affiche les images avec bounding boxes."""
    model.eval()

    cols = 3
    rows = (num_images + 2) // 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axes = axes.flatten()

    indices = np.random.choice(len(dataset), min(num_images, len(dataset)), replace=False)

    for idx, ax in enumerate(axes):
        if idx >= len(indices):
            ax.axis('off')
            continue

        img, cls_target, reg_target = dataset[indices[idx]]

        # Prediction
        cls_pred, reg_pred = model(img.unsqueeze(0).to(device))
        cls_pred = (torch.sigmoid(cls_pred[0]) > 0.5).cpu()
        reg_pred = reg_pred[0].cpu()

        # Afficher image
        img_np = denormalize(img)
        ax.imshow(img_np)
        h, w = img_np.shape[:2]

        # Dessiner boxes
        for c in range(2):
            # Ground Truth (vert pointille)
            if cls_target[c] == 1:
                x_c, y_c, bw, bh = reg_target[c].numpy()
                x1, y1 = (x_c - bw/2) * w, (y_c - bh/2) * h
                rect = patches.Rectangle((x1, y1), bw*w, bh*h,
                    linewidth=2, edgecolor='green', facecolor='none', linestyle='--')
                ax.add_patch(rect)
                ax.text(x1, y1-5, f'GT {CLASSES[c]}', color='green', fontsize=9)

            # Prediction (rouge/bleu)
            if cls_pred[c]:
                x_c, y_c, bw, bh = reg_pred[c].numpy()
                x1, y1 = (x_c - bw/2) * w, (y_c - bh/2) * h
                rect = patches.Rectangle((x1, y1), bw*w, bh*h,
                    linewidth=2, edgecolor=COLORS[c], facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1 + bh*h + 10, f'Pred {CLASSES[c]}', color=COLORS[c], fontsize=9)

        ax.set_title(f"Image {indices[idx]}")
        ax.axis('off')

    plt.tight_layout()
    if output_dir:
        from pathlib import Path
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'predictions.png'
    else:
        output_path = 'predictions.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Sauvegarde: {output_path}")


def calculate_iou(box1, box2):
    """
    Calcule l'IoU (Intersection over Union) entre deux boîtes.
    """
    # Convertir en numpy si nécessaire
    if torch.is_tensor(box1):
        box1 = box1.numpy()
    if torch.is_tensor(box2):
        box2 = box2.numpy()
    
    # Convertir de [x_c, y_c, w, h] vers [x1, y1, x2, y2]
    def to_corners(box):
        x_c, y_c, w, h = box
        x1 = x_c - w/2
        y1 = y_c - h/2
        x2 = x_c + w/2
        y2 = y_c + h/2
        return [x1, y1, x2, y2]
    
    box1 = to_corners(box1)
    box2 = to_corners(box2)
    
    # Calculer l'intersection
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Aire de l'intersection
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height
    
    # Aire de chaque boîte
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Aire de l'union
    union_area = box1_area + box2_area - inter_area
    
    # IoU
    if union_area == 0:
        return 0.0
    
    iou = inter_area / union_area
    return iou


def print_evaluation_results(results):
    """Affiche les résultats d'évaluation de manière lisible."""
    print("\n" + "="*60)
    print("📊 RÉSULTATS DE L'ÉVALUATION")
    print("="*60)
    
    print(f"\n📈 Métriques globales :")
    print(f"   • Accuracy classification : {results['cls_accuracy']:.4f} ({results['cls_accuracy']*100:.2f}%)")
    print(f"   • IoU moyen               : {results['mean_iou']:.4f} ({results['mean_iou']*100:.2f}%)")
    print(f"   • Seuil IoU               : {results['iou_threshold']}")
    print(f"   • Images évaluées         : {results['total_images']}")
    
    # Par classe
    print(f"\n📊 Résultats par classe :")
    print("-"*60)
    
    for c, stats in results['per_class'].items():
        print(f"\n  {CLASSES[c]} :")
        print(f"     • Précision  : {stats['precision']:.4f} ({stats['precision']*100:.2f}%)")
        print(f"     • Recall     : {stats['recall']:.4f} ({stats['recall']*100:.2f}%)")
        print(f"     • F1-Score   : {stats['f1']:.4f} ({stats['f1']*100:.2f}%)")
        print(f"     • IoU moyen  : {stats['mean_iou']:.4f}")
        print(f"     • TP={stats['tp']}, FP={stats['fp']}, FN={stats['fn']}")
    
    # Distribution des IoU
    ious = results['all_ious']
    if ious:
        print(f"\n📊 Distribution des IoU :")
        print(f"   • Min       : {min(ious):.4f}")
        print(f"   • Max       : {max(ious):.4f}")
        print(f"   • Médiane   : {np.median(ious):.4f}")
        print(f"   • Écart-type: {np.std(ious):.4f}")
    
    print("="*60 + "\n")


def load_model(model_path, num_classes=2):
    """Charge le modèle sauvegardé."""
    model = MultiObjectDetector(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Modèle chargé: {model_path}")
    return model


if __name__ == "__main__":
    print("Evaluation complete du modele sur le test set\n")

    # Charger les données
    import configparser
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parents[2]
    CONFIG = configparser.ConfigParser()
    CONFIG.read(PROJECT_ROOT / 'config.ini')
    DATASET_DIR = str(PROJECT_ROOT / CONFIG.get('data', 'DATASET_DIR'))
    MODEL_PATH = PROJECT_ROOT / CONFIG.get('cnn', 'TRAINED_MODEL')
    RUNS_DIR = PROJECT_ROOT / CONFIG.get('cnn', 'RUNS_DIR')

    _, _, test_loader = get_loaders(data_dir=DATASET_DIR, batch_size=BATCH_SIZE)
    test_dataset = test_loader.dataset

    # Charger le modèle
    model = load_model(MODEL_PATH)
    img, cls_target, reg_target = test_dataset[0]
    _, reg_pred = model(img.unsqueeze(0).to(device))
    print("Pred boxes:", reg_pred[0].cpu().detach().numpy())
    print("GT boxes:", reg_target.detach().numpy())

    # Evaluer
    test_results = evaluate_all_dataset(model, test_dataset, iou_threshold=0.5)
    print_evaluation_results(test_results)
    plot_predictions(model, test_dataset, num_images=9, output_dir=RUNS_DIR)
