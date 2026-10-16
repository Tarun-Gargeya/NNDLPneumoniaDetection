# Pediatric Pneumonia Detection: Do Correct-Looking Heatmaps Mean Correct Reasoning?

An EfficientNet-B3 classifier for pneumonia in pediatric chest X-rays, with Grad-CAM explanations, a recall-first decision threshold and a Streamlit demo. On top of this standard pipeline, the project asks one question: **if a model's Grad-CAM heatmap points at the lungs, has it stopped using shortcuts?**

> Course project for Neural Networks & Deep Learning. Not for clinical use.

---

## 1. Problem

Pneumonia is a leading cause of death in children under five. Diagnosis depends on a radiologist reading a chest X-ray, and radiologists are scarce in many clinics. A deep-learning triage assistant could flag likely pneumonia cases for review.

The standard benchmark is the **Kermany pediatric chest X-ray dataset** (5,856 images, Guangzhou Women & Children's Medical Center, via [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)). Models on this dataset routinely report 95–99% accuracy, and many papers add a Grad-CAM heatmap as evidence that the model is "explainable".

## 2. Novelty

**The dataset contains a shortcut.** Normal and Pneumonia images were captured or exported differently. Pneumonia images are smaller (log pixel-area 13.7 vs 14.6) and much wider (aspect ratio 1.52 vs 1.25). A logistic regression that sees **only image size, shape and brightness, and no lung content**, predicts the label with **test AUC 0.926**.

**What existing work does:** shows Grad-CAM heatmaps on the lungs, or trains the model with an extra loss that pushes the heatmap onto the lungs, and treats that as evidence of trustworthy reasoning.

**Our contribution:** a controlled comparison of four models that measures *where the model looks* separately from *what information the model uses*:

1. **Heatmap location:** the share of Grad-CAM energy inside a lung mask.
2. **Whether the shortcut information is still present:** a linear probe that tries to recover the original image size and aspect ratio from the network's features.
3. **Whether the model relies on the shortcut:** AUC on a **matched-acquisition test subset**, where Normal and Pneumonia images have the same size and aspect-ratio distribution. There, metadata alone drops to AUC 0.569, near chance.
4. **A fix attempt:** an adversarial "shortcut-blind" head behind a gradient reversal layer, combined with augmentation that randomizes acquisition properties.

Each building block (Grad-CAM guidance, gradient reversal, linear probes) exists in the literature. We did not find prior work that combines them to test whether heatmap guidance removes the acquisition shortcut on this dataset.

## 3. Key findings

| Model | Test AUC | AUC, shortcut removed | Grad-CAM inside lungs | Size recoverable (probe R²) | Aspect ratio recoverable (probe R²) |
|---|---|---|---|---|---|
| Metadata only (no image) | 0.926 | 0.569 | – | – | – |
| Baseline EfficientNet-B3 | **0.995** | **0.981** | 0.27 | 0.75 | 0.67 |
| + Grad-CAM lung guidance | 0.994 | 0.977 | 0.985 | 0.76 | 0.68 |
| + guidance + acquisition augmentation | 0.993 | 0.975 | **0.990** | 0.76 | 0.69 |
| + guidance + augmentation + adversary (ours) | 0.983 | 0.959 | 0.957 | **0.71** | **0.64** |

Patient-level test set: 878 images. Matched-acquisition subset: 264 images. One training run per model.

1. **The baseline's heatmaps are mostly outside the lungs.** Only 27% of its Grad-CAM energy falls inside the lung fields. On several pneumonia X-rays it focuses on the **"R" laterality marker in the image corner** (see `results/figures/gradcam_gallery.png`). It classifies well, but its explanation would not convince a radiologist.
2. **Grad-CAM guidance fixes the heatmaps at no accuracy cost.** Lung-focused energy rises from 27% to 98.5%, and test AUC stays at 0.994.
3. **The heatmaps look right, but the shortcut information remains.** After guidance, the image's original size and aspect ratio are just as recoverable from the features as before (R² 0.75 → 0.76 for size, 0.67 → 0.68 for aspect ratio). **A correct-looking heatmap does not show that shortcut information was removed.** This is the project's main finding.
4. **The adversarial fix works only partly.** It lowers how much shortcut information remains (aspect ratio R² 0.69 → 0.64), and it halves the within-class correlation between the model's pneumonia score and image size (0.29 → 0.15). But it costs accuracy (test AUC 0.983) and does not improve performance on the matched subset. Removing shortcut information without losing useful signal is a trade-off, not a free gain.
5. **The CNNs depend on the shortcut much less than the metadata classifier does.** When the shortcut is removed, every CNN loses only about 1.5–2.5 AUC points, while the metadata classifier falls to near chance. The CNNs *encode* the shortcut but get most of their accuracy from real image content.

## 4. How it works

### Pipeline

```
Kaggle X-rays ──► prepare.py ──────────► shortcut_check.py ──► train.py ×4 ──► evaluate.py ──► app.py
                  • remove duplicates    • metadata-only AUC     • two-phase     • metrics       • demo
                  • patient-level split  • matched test subset    • extra losses  • probes
                  • 256×256 cache                                   per variant    • figures
                  • lung masks (PSPNet)
```

### Data (`src/prepare.py`)
- 5,856 unique images from 3,118 patients. The Kaggle zip contains a duplicate folder tree, which is removed.
- **Patient-level 70/15/15 split**, stratified by class: train 4,121 / val 857 / test 878 images. No child's X-rays appear in more than one split. Kaggle's original split leaks patients between train and test, which inflates accuracy.
- Acquisition metadata (size, aspect ratio, brightness, contrast) is measured on the **original** image before resizing.
- **Lung masks** come from a pretrained PSPNet in `torchxrayvision`. Median lung coverage is 36%; 6 images where segmentation failed fall back to a full mask.

### Model (`src/common.py`)
The architecture from the project proposal:

```
224×224 X-ray ─► EfficientNet-B3 (ImageNet) ─► 7×7×1536 ─► global avg pool ─► Dense 256 + ReLU ─► Dropout 0.5 ─► 2 logits
                                                                   │
                                                                   └─► gradient reversal ─► shortcut adversary (predicts size, aspect)   [ours only]
```

### Training (`src/train.py`)
| | Phase 1: feature extraction | Phase 2: fine-tuning |
|---|---|---|
| Trainable | Classifier head only | Head + last two EfficientNet stages and head convolution |
| Optimizer | AdamW, lr 1e-3 | AdamW, lr 5e-5 |
| Epochs | up to 6 | up to 10 |

- Weighted cross-entropy with **Normal ×2.74**. Normal is the minority class: 1,101 vs 3,020 training images.
- ReduceLROnPlateau (×0.1), early stopping on validation AUC (patience 5), batch size 32, bfloat16 mixed precision.
- Augmentation runs on the GPU and is applied identically to the image and its lung mask: flip, ±15° rotation, 0.9–1.1 zoom, brightness jitter.

**The four variants** differ only in their extra losses and augmentation:

| Variant | Grad-CAM lung loss | Acquisition augmentation | Shortcut adversary |
|---|---|---|---|
| `baseline` | | | |
| `camguide` | ✓ | | |
| `camguide_aug` | ✓ | ✓ | |
| `ours` | ✓ | ✓ | ✓ |

- **Grad-CAM lung loss:** Grad-CAM for the true class is computed in a differentiable way during training. The loss is the share of its energy outside the lung mask.
- **Acquisition augmentation:** random aspect-ratio stretch (±35%), gamma, contrast, resolution reduction and noise, so the original size and shape stop being reliable cues.
- **Shortcut adversary:** a small head predicts the original image's size and aspect ratio from the pooled features. A gradient reversal layer, ramped up over phase 2, pushes the backbone to remove that information.

### Evaluation (`src/evaluate.py`)
Reports AUC, accuracy, precision, recall and F1, plus recall and specificity at a **recall-first threshold**: the threshold is chosen on validation to give ≥95% recall, so missed pneumonia cases are minimized. It also reports AUC on the matched-acquisition subset, the lung-energy score of Grad-CAM, the linear-probe R² values, and the within-class rank correlation between the model's score and the acquisition properties.

## 5. Setup and running

**Requirements:** Windows or Linux, Python 3.12, an NVIDIA GPU (tested on an RTX 5070 Ti Laptop, 12 GB) and a Kaggle account.

```bash
# 1. Environment
python -m venv .venv
source .venv/Scripts/activate      # Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 2. Kaggle credentials
cp .env.example .env               # then fill in KAGGLE_USERNAME + KAGGLE_KEY, or KAGGLE_API_TOKEN

# 3. Data (about 1.2 GB download; preparation about 3 minutes)
python src/download_data.py
PYTHONIOENCODING=utf-8 python src/prepare.py

# 4. Metadata check, train all 4 variants, evaluate (about 10 minutes on the GPU)
PYTHONIOENCODING=utf-8 ./run_all.sh
```

The commands assume Git Bash on Windows or a regular shell on Linux. `PYTHONIOENCODING=utf-8` avoids a console encoding error in torchxrayvision's download progress bar on Windows. Each training run takes about 1.5 minutes.

Trained weights (`results/<variant>/model.pth`, about 45 MB each) are not tracked in git, so run step 4 before starting the demo.

### Demo

```bash
streamlit run app.py
```

Upload a chest X-ray; sample test-set images the models never trained on are in `demo_samples/`. The app shows **NORMAL / PNEUMONIA**, the pneumonia probability and a Grad-CAM overlay. You can switch between the four models to compare their heatmaps. The threshold slider shows the recall-first trade-off: lowering it catches more pneumonia cases at the cost of more false alarms.

## 6. Repository layout

```
├── app.py                  Streamlit demo
├── run_all.sh              shortcut check → train 4 variants → evaluate
├── requirements.txt
├── .env.example            template for Kaggle credentials (copy to .env)
├── src/
│   ├── download_data.py    Kaggle download
│   ├── prepare.py          duplicate removal, patient split, metadata, resize cache, lung masks
│   ├── shortcut_check.py   metadata-only classifier, matched-acquisition subset
│   ├── common.py           model, GPU augmentation, differentiable Grad-CAM, gradient reversal
│   ├── train.py            two-phase training of one variant
│   └── evaluate.py         metrics, probes, figures
├── demo_samples/           6 held-out test X-rays for the demo
├── data/                   raw + processed data (git-ignored)
└── results/
    ├── metrics.csv / metrics.md / shortcut_check.json
    ├── <variant>/model.pth, history.csv
    └── figures/  roc.png · summary.png · gradcam_gallery.png · confusion.png
```

## 7. Changes from the original proposal

| Proposal | Implemented | Why |
|---|---|---|
| Keras/TensorFlow | PyTorch + timm | TensorFlow has no native GPU support on Windows; custom losses and gradient reversal are simpler in PyTorch |
| Random 70/15/15 split | Patient-level 70/15/15 split | Prevents the same child appearing in both train and test |
| Class weight pneumonia: 4 | Normal ×2.74 | Pneumonia is the majority class; the original weighting made the imbalance worse |
| "[0,1] with ImageNet mean/std" | ImageNet mean/std normalization | The original description was contradictory |
| Fine-tune "top 20 layers" | Last two EfficientNet stages + head convolution | The closest equivalent in PyTorch |
| Fine-tune lr 1e-5, 10 epochs | lr 5e-5, up to 10 epochs | Converges within the shorter schedule |

## 8. Limitations

- **One run per model.** The differences between the three guided models (around 0.002–0.01 AUC) are within run-to-run noise. Results with several random seeds would be needed before drawing fine-grained conclusions.
- **Single dataset.** There is no external validation on another hospital's data, which is where a shortcut-free model would show its real advantage.
- **Small matched subset.** 264 images, and metadata alone still reaches AUC 0.569 there rather than exactly 0.5.
- **The adversary's strength was not tuned.** Its loss weight is fixed at λ = 1. A sweep might find a better trade-off between removing the shortcut and keeping accuracy.
- **Lung masks come from a model trained on adult X-rays** and are less accurate on some pediatric images.

## 9. References

- Kermany et al., *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*, Cell, 2018.
- Tan & Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, ICML, 2019.
- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV, 2017.
- Ganin & Lempitsky, *Unsupervised Domain Adaptation by Backpropagation* (gradient reversal), ICML, 2015.
- Ross, Hughes & Doshi-Velez, *Right for the Right Reasons*, IJCAI, 2017.
- DeGrave, Janizek & Lee, *AI for radiographic COVID-19 detection selects shortcuts over signal*, Nature Machine Intelligence, 2021.
- Rajpurkar et al., *CheXNet*, arXiv, 2017.
- Cohen et al., *TorchXRayVision: A library of chest X-ray datasets and models*, MIDL, 2022.
