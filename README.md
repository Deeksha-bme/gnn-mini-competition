# GNN Mini-Competition – G4RS

## Overview
This repository contains my work for the G4RS mini-competition on Graph Neural Networks (GNNs).
The objective is to experiment with graph-based models and analyze their performance in a research-oriented setting.

## Task Description
This task focuses on applying Graph Neural Networks to a graph-based learning problem provided as part of the G4RS mini-competition.
The goal is to explore how node features and graph structure can be leveraged for predictive modeling.

## Dataset
- Source: Provided by the G4RS program / mini-competition
- Description: A graph dataset consisting of nodes, edges, and associated features
- Preprocessing: Standard data loading and splitting into training, validation, and test sets

## Methodology
- Baseline Model: Graph Convolutional Network (GCN)
- Framework: PyTorch and PyTorch Geometric
- Training Setup:
  - Optimizer: Adam
  - Loss Function: Cross-Entropy Loss
  - Epochs: 100
  - Evaluation Metric: F1-score / Accuracy

## Results

### 🏆 Leaderboard

| Rank | Name     | Model | Metric | Score | Date |
|------|----------|-------|--------|-------|------|
| 1    | Deeksha  | GCN   | Accuracy | 0.81 | Jan 2026 |


*(The leaderboard will be updated as experiments and results improve.)*

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
