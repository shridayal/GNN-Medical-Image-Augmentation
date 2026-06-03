# GNN-Guided Generative AI for Medical Image Augmentation

This repository now defines a focused blueprint for a **GNN-guided generative augmentation pipeline** for medical imaging.

## Objective

Improve downstream model performance and robustness on limited/imbalanced medical imaging datasets by generating clinically plausible synthetic samples using graph-informed conditioning.

## Core Idea

1. Represent patient, modality, anatomical, and pathology relationships as a graph.
2. Learn node/edge embeddings with a Graph Neural Network (GNN).
3. Condition a generative model (for example, diffusion or GAN-based) on those embeddings.
4. Generate controlled synthetic images and labels.
5. Filter generated samples using quality/safety checks before training.

## Minimal System Design

### Inputs

- Medical images (CT/MRI/X-ray/Ultrasound)
- Clinical metadata (age, sex, diagnosis, modality, acquisition context)
- Optional segmentation masks / region annotations

### Graph Construction

- **Nodes**: patient cohort groups, modalities, pathology classes, anatomy regions
- **Edges**: clinically meaningful relations (co-occurrence, progression, modality compatibility)
- **Node/edge features**: normalized metadata and prior statistics

### GNN Module

- Learns latent graph representation capturing structural clinical dependencies
- Exports conditioning vectors for each augmentation target

### Generative Module

- Uses GNN conditioning vectors to guide synthesis
- Supports class-conditional and structure-aware generation
- Produces image (+ optional mask) pairs

### Quality & Safety Gate

- Distribution checks (feature-space distance / FID-like proxy)
- Clinical consistency checks against conditioning labels
- Optional segmentation plausibility constraints

## Training Workflow

1. Preprocess images + metadata
2. Build medical relationship graph
3. Train GNN encoder
4. Train conditional generator with GNN embeddings
5. Generate candidate augmentations
6. Apply quality filters
7. Train downstream diagnostic/segmentation model on real + synthetic data
8. Evaluate against non-augmented and baseline augmentation setups

## Evaluation Targets

- Classification: AUROC, F1, sensitivity/specificity
- Segmentation: Dice, IoU, Hausdorff distance
- Robustness: minority-class and low-data regime improvements
- Safety: reduction of unrealistic artifacts in accepted synthetic samples

## Scope of This Repository

The repository currently captures the project definition and implementation direction for building a GNN-guided generative augmentation system in a modular way.
