# Luminara: CRF Framework for AI Investment Decisions

Luminara is a probabilistic framework that leverages Conditional Random Fields (CRF) to predict the worthiness of investing in AI products. By modeling the complex interdependencies between market, development, and customer-centric variables, Luminara provides data-driven investment recommendations with comprehensive explanations.

## Overview

Predicting the success of AI investments requires understanding complex relationships between multiple variables. Luminara uses a factor graph representation with carefully designed clique structures to capture these dependencies:

### Key Variables

- **Market Variables**:
  - Market size (M): small (0), medium (1), large (2)
  - Competitor penetration (C): low (0), medium (1), high (2)
  - Growth rate (G): slow (0), moderate (1), rapid (2)
  - Revenue potential (RP): low (0), medium (1), high (2)

- **Development Variables**:
  - Bug density (B): low (0), medium (1), high (2)
  - Development cost (DC): low (0), medium (1), high (2)
  - Development time (DT): short (0), medium (1), long (2)
  - Feature completion (FC): incomplete (0), partial (1), complete (2)

- **Customer Variables**:
  - Customer acquisition cost (CAC): low (0), medium (1), high (2)
  - Return rate (R): low (0), medium (1), high (2)
  - Average time to action (ATA): quick (0), medium (1), slow (2)

- **Target Variable**:
  - Investment worthiness (Y): not worthy (0), worthy (1)

### Clique Structure

Luminara organizes variables into interconnected cliques to model domain-specific dependencies:

1. **Market Clique**: M, C, G, RP
2. **Development Clique**: B, DC, DT, FC
3. **Customer Clique**: CAC, R, ATA
4. **Bridge Cliques**: {RP, Y}, {DC, Y}, {R, Y}

The bridge cliques directly connect to the target variable and have the strongest influence on investment worthiness predictions.

## Technical Implementation

Luminara implements state-of-the-art probabilistic graphical model techniques:

- **Model Representation**: Factor graphs for efficient inference
- **Parameter Estimation**: L-BFGS optimization for learning model parameters
- **Inference Algorithm**: Belief propagation for probabilistic predictions
- **Decision Strategy**: Maximum a Posteriori (MAP) inference for final investment decisions
- **Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1, feature importance, and calibration

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Luminara.git
cd Luminara

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Generating Synthetic Data

```bash
python -m src.data.dataset
```

#### Training a Model

```bash
python results/train_model.py --dataset synthetic
```

#### Making Investment Decisions

```bash
python examples/investment_decision_demo.py
```

## Project Structure

```
Luminara/
├── README.md                     # Project documentation
├── requirements.txt              # Dependencies
├── setup.py                      # Package setup
├── config/                       # Configuration files
│   └── model_config.yaml         # Model configuration parameters
├── data/
│   ├── processed/                # Processed data ready for modeling
│   └── synthetic/                # Synthetic generated data
├── examples/
│   └── investment_decision_demo.py # Interactive demo application
├── notebooks/                    # Jupyter notebooks
├── results/
│   ├── eval/                     # Evaluation results
│   │   └── synthetic/            # Evaluation results for synthetic data
│   ├── logs/                     # Training and inference logs
│   │   └── synthetic/            # Logs for synthetic data runs
│   ├── models/                   # Saved model files
│   │   └── synthetic/            # Models trained on synthetic data
│   └── train_model.py            # Model training script
└── src/                          # Source code
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   ├── dataset.py            # Dataset generation and handling
    │   └── synthetic.py          # Synthetic data generation utilities
    ├── models/
    │   ├── __init__.py
    │   ├── cliques.py            # Clique definition and factor creation
    │   ├── crf.py                # Main CRF model implementation
    │   ├── evaluation.py         # Model evaluation metrics
    │   ├── inference.py          # Belief propagation and inference
    │   └── train.py              # Model training functions
    └── utils/
        ├── __init__.py
        ├── logger.py             # Logging utilities
        └── visualization.py      # Factor graph visualization
```

## Features

- **Data Generation**: Create realistic synthetic data with proper variable dependencies
- **Model Training**: Train CRF models with optimized parameter estimation
- **Inference**: Make probabilistic predictions about investment worthiness
- **Evaluation**: Assess model performance with comprehensive metrics
- **Decision Reports**: Generate detailed investment decision reports with explanations
- **Visualization**: Visualize factor graphs and evaluation results
- **Interactive Demo**: Test the model with custom investment scenarios

## Advanced Features

- **Feature Importance Analysis**: Understand which variables most strongly influence predictions
- **Cross-Validation**: Ensure model stability and generalization
- **Evidence Influence Analysis**: See how changing evidence affects predictions
- **Probability Calibration**: Ensure predicted probabilities reflect true likelihoods
- **Customizable Cliques**: Adapt the model structure to different investment domains