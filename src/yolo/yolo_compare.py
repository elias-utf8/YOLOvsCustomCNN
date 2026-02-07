"""
=============================================================================
  YOLO Benchmark — Train, Test, Log & Compare
=============================================================================
  1. Configure les paramètres ci-dessous
  2. Lance : python yolo_compare.py
  3. Résultats ajoutés dans data/yolo/results/benchmark_results.txt
  4. Graphiques de comparaison mis à jour dans data/yolo/results/
=============================================================================
"""
import configparser
from pathlib import Path
from datetime import datetime
import json
import time
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Config
PROJECT_ROOT = Path(__file__).parents[2]
CONFIG = configparser.ConfigParser()
CONFIG.read(PROJECT_ROOT / 'config.ini')

# Chemins depuis config.ini
DATASET_DIR = PROJECT_ROOT / CONFIG.get('data', 'DATASET_DIR')
DATASET_YAML = PROJECT_ROOT / CONFIG.get('data', 'DATASET_YAML')
TEST_IMAGES_DIR = DATASET_DIR / 'images' / 'test'
PRETRAINED_MODEL = PROJECT_ROOT / CONFIG.get('yolo', 'PRETRAINED_MODEL')
RUNS_DIR = PROJECT_ROOT / CONFIG.get('yolo', 'RUNS_DIR')

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    PARAMÈTRES À CONFIGURER                                
# ╚═══════════════════════════════════════════════════════════════════════════╝

MODEL_NAME      = PRETRAINED_MODEL.name
DATA_YAML       = str(DATASET_YAML)

EPOCHS          = 50
BATCH_SIZE      = 2
IMG_SIZE        = 224
PATIENCE        = 10                    # Early stopping (0 = désactivé)
DEVICE          = 0                    # 0 = GPU, "cpu" = CPU

CONF_THRESHOLD  = 0.5                   # Seuil de confiance pour les predictions visuelles
NB_PRED_IMAGES  = 5                     # Nombre d'images de test a predire visuellement

RESULTS_DIR     = str(RUNS_DIR / "results")
PROJECT_DIR     = str(RUNS_DIR)

EXPERIMENT_NOTE = ""                    # Note libre pour identifier l'expérience

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         FIN DE LA CONFIGURATION                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


def get_run_name():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = MODEL_NAME.replace(".pt", "")
    return f"{base}_e{EPOCHS}_b{BATCH_SIZE}_img{IMG_SIZE}_{ts}"


def train(run_name):
    print(f"\n{'='*60}")
    print(f"   ENTRAÎNEMENT — {run_name}")
    print(f"{'='*60}\n")

    model = YOLO(str(PRETRAINED_MODEL))

    start = time.time()
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name=run_name,
        patience=PATIENCE,
        device=DEVICE,
        project=PROJECT_DIR,
    )
    train_time = time.time() - start

    print(f"\n Entraînement terminé en {train_time:.1f}s ({train_time/60:.1f} min)")
    return results, train_time


def test(run_name):
    print(f"\n{'='*60}")
    print(f"   ÉVALUATION — {run_name}")
    print(f"{'='*60}\n")

    best_weights = Path(PROJECT_DIR) / run_name / "weights" / "best.pt"
    if not best_weights.exists():
        print(f"  Poids non trouvés : {best_weights}")
        return None, 0

    model = YOLO(str(best_weights))

    start = time.time()
    metrics = model.val(data=DATA_YAML, split="test")
    test_time = time.time() - start

    results = {
        "mAP50":      round(metrics.box.map50, 4),
        "mAP50_95":   round(metrics.box.map, 4),
        "precision":  round(metrics.box.mp, 4),
        "recall":     round(metrics.box.mr, 4),
    }

    print(f"\n Résultats TEST SET :")
    print(f"   mAP@0.5      : {results['mAP50']:.4f}")
    print(f"   mAP@0.5:0.95 : {results['mAP50_95']:.4f}")
    print(f"   Precision     : {results['precision']:.4f}")
    print(f"   Recall        : {results['recall']:.4f}")
    print(f"   Temps test    : {test_time:.1f}s")

    # Prédictions visuelles
    if TEST_IMAGES_DIR.exists():
        images = sorted(
            list(TEST_IMAGES_DIR.glob("*.jpg")) + list(TEST_IMAGES_DIR.glob("*.png"))
        )[:NB_PRED_IMAGES]
        if images:
            print(f"\n📸 Prédiction sur {len(images)} images...")
            for img in images:
                model.predict(
                    source=str(img),
                    conf=CONF_THRESHOLD,
                    save=True,
                    project=PROJECT_DIR,
                    name=f"{run_name}_predictions",
                    exist_ok=True,
                )
                print(f" {img.name}")

    return results, test_time


def save_results(run_name, metrics, train_time, test_time):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    txt_path  = Path(RESULTS_DIR) / "benchmark_results.txt"
    json_path = Path(RESULTS_DIR) / "benchmark_results.json"

    entry = {
        "run_name":     run_name,
        "timestamp":    datetime.now().isoformat(),
        "model":        MODEL_NAME,
        "epochs":       EPOCHS,
        "batch_size":   BATCH_SIZE,
        "img_size":     IMG_SIZE,
        "patience":     PATIENCE,
        "train_time_s": round(train_time, 1),
        "test_time_s":  round(test_time, 1),
        "note":         EXPERIMENT_NOTE,
        **metrics,
    }

    # --- Fichier texte lisible ---
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"  {run_name}\n")
        f.write(f"  {entry['timestamp']}\n")
        if EXPERIMENT_NOTE:
            f.write(f"  Note: {EXPERIMENT_NOTE}\n")
        f.write(f"{'='*70}\n")
        f.write(f"  Modèle     : {MODEL_NAME}\n")
        f.write(f"  Epochs     : {EPOCHS}  |  Batch : {BATCH_SIZE}  |  ImgSize : {IMG_SIZE}\n")
        f.write(f"  Patience   : {PATIENCE}\n")
        f.write(f"  ─────────────────────────────────\n")
        f.write(f"  mAP@0.5      : {metrics['mAP50']:.4f}\n")
        f.write(f"  mAP@0.5:0.95 : {metrics['mAP50_95']:.4f}\n")
        f.write(f"  Precision    : {metrics['precision']:.4f}\n")
        f.write(f"  Recall       : {metrics['recall']:.4f}\n")
        f.write(f"  ─────────────────────────────────\n")
        f.write(f"  Temps train  : {train_time:.1f}s ({train_time/60:.1f} min)\n")
        f.write(f"  Temps test   : {test_time:.1f}s\n")
        f.write(f"{'='*70}\n\n")

    print(f"\n Résultats ajoutés dans : {txt_path}")

    # --- Fichier JSON (pour les graphiques) ---
    all_results = []
    if json_path.exists():
        with open(json_path, "r") as f:
            all_results = json.load(f)
    all_results.append(entry)
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    return all_results


def generate_plots(all_results):
    """Génère les graphiques de comparaison entre toutes les expériences."""
    if len(all_results) < 1:
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Labels ---
    names = []
    for r in all_results:
        label = r.get("note") if r.get("note") else r["run_name"]
        if len(label) > 30:
            label = label[:27] + "..."
        names.append(label)

    map50     = [r["mAP50"] for r in all_results]
    map50_95  = [r["mAP50_95"] for r in all_results]
    precision = [r["precision"] for r in all_results]
    recall    = [r["recall"] for r in all_results]
    train_t   = [r.get("train_time_s", 0) / 60 for r in all_results]

    x = np.arange(len(names))

    # ── 1. Barres groupées : 4 métriques ──
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 2.5), 6))
    w = 0.2
    ax.bar(x - 1.5*w, map50,     w, label="mAP@0.5",      color="#4C72B0")
    ax.bar(x - 0.5*w, map50_95,  w, label="mAP@0.5:0.95", color="#55A868")
    ax.bar(x + 0.5*w, precision, w, label="Precision",     color="#C44E52")
    ax.bar(x + 1.5*w, recall,    w, label="Recall",        color="#8172B2")
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Comparaison des métriques YOLO")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(RESULTS_DIR) / "compare_metrics.png", dpi=150)
    plt.close()

    # ── 2. Temps d'entraînement ──
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 2), 5))
    bar_colors = plt.cm.Set2(np.linspace(0, 1, max(len(names), 2)))
    bars = ax.bar(x, train_t, color=bar_colors[:len(names)], edgecolor="white")
    ax.bar_label(bars, fmt="%.1f min", fontsize=9, padding=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Temps (minutes)")
    ax.set_title("Temps d'entraînement par expérience")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(RESULTS_DIR) / "compare_time.png", dpi=150)
    plt.close()

    # ── 3. Radar chart (>= 2 expériences) ──
    if len(all_results) >= 2:
        categories = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall"]
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        rc = plt.cm.tab10(np.linspace(0, 1, min(len(all_results), 10)))

        for i, r in enumerate(all_results):
            vals = [r["mAP50"], r["mAP50_95"], r["precision"], r["recall"]]
            vals += vals[:1]
            ax.plot(angles, vals, "o-", linewidth=2, label=names[i], color=rc[i % 10])
            ax.fill(angles, vals, alpha=0.1, color=rc[i % 10])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title("Radar — Comparaison des expériences", y=1.08)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
        plt.tight_layout()
        plt.savefig(Path(RESULTS_DIR) / "compare_radar.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ── 4. Évolution au fil des expériences (>= 3) ──
    if len(all_results) >= 3:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        exp_x = range(1, len(all_results) + 1)

        ax1.plot(exp_x, map50,     "o-", label="mAP@0.5",      linewidth=2)
        ax1.plot(exp_x, map50_95,  "s-", label="mAP@0.5:0.95", linewidth=2)
        ax1.plot(exp_x, precision, "^-", label="Precision",     linewidth=2)
        ax1.plot(exp_x, recall,    "D-", label="Recall",        linewidth=2)
        ax1.set_xlabel("Expérience #")
        ax1.set_ylabel("Score")
        ax1.set_title("Évolution des métriques")
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_xticks(list(exp_x))

        ax2.plot(exp_x, train_t, "o-", color="#E07B39", linewidth=2, markersize=8)
        ax2.set_xlabel("Expérience #")
        ax2.set_ylabel("Temps (minutes)")
        ax2.set_title("Évolution du temps d'entraînement")
        ax2.grid(alpha=0.3)
        ax2.set_xticks(list(exp_x))

        plt.tight_layout()
        plt.savefig(Path(RESULTS_DIR) / "compare_evolution.png", dpi=150)
        plt.close()

    print(f"📊 Graphiques sauvegardés dans : {RESULTS_DIR}/")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                              MAIN                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    total_start = time.time()

    run_name = get_run_name()

    # 1. Train
    _, train_time = train(run_name)

    # 2. Test
    metrics, test_time = test(run_name)

    if metrics:
        # 3. Sauvegarder les résultats
        all_results = save_results(run_name, metrics, train_time, test_time)

        # 4. Générer les graphiques de comparaison
        generate_plots(all_results)

    total = time.time() - total_start
    print(f"\n Temps total : {total:.1f}s ({total/60:.1f} min)")
    print(f" Terminé ! Consultez {RESULTS_DIR}/ pour les résultats et graphiques.")
