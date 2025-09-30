# Machine Learning Models

This directory contains trained machine learning models.  
Each model is documented below with details about its purpose, training setup, and usage.

---

## 📂 Directory Structure
```
models/
├── model_freq
    ├──Iter_Number
        ├──model.pkl
        ├──metrics.pkl
├── model_amp
    ├──Iter_Number
        ├──model.pkl
        ├──metrics.pkl
├── model_freq_update
    ├──Iter_Number
        ├──model.pkl
        ├──metrics.pkl
├── model_freq_withPrior
    ├──Iter_Number
        ├──model.pkl
        ├──metrics.pkl
└── README.md
```

---

## 📑 Model Descriptions

### 1. `model_freq.model.pkl`

- **Type:** LNN trained on S-curves
- **Training Regime:** Forcing Amplitude - 0.1 to 2.0 N, Forcing Frequency - 10.0 to 24.0 Hz. See `contparameters.json` for numerical continuations parameters.
- **Input Features:**  
  - `q`: Modal Displacements  
  - `qdot`: Modal Velocities  
- **Output:**
  - `qddot`: Modal Accelerations
- **Training Dataset:** S-curves obtained from numerical continuation. See [data](../Conx/modal_freq/data.pkl)
- **Model Files:** [MDOF_LNN.py](../models/MDOF_LNN.py)
- **Performance Metrics:**  
  - MSE
- **Dependencies:**  
  - `jax`, `numpy`, `scipy`, `diffrax`
- **Usage Example:**
  - See `./models/model_data_ext.ipynb`
  
### 2. `model_freq_update.model.pkl`

- **Type:** LNN trained on S-curves
- **Training Regime:** Forcing Amplitude - 0.1 to 8.0 N, Forcing Frequency - 10.0 to 24.0 Hz. See `contparameters.json` for numerical continuations parameters.
- **Input Features:**  
  - `q`: Modal Displacements  
  - `qdot`: Modal Velocities  
- **Output:**
  - `qddot`: Modal Accelerations
- **Training Dataset:** S-curves obtained from numerical continuation. . See [data](../Conx/modal_freq_update/data.pkl)
- **Model Files:** [MDOF_LNN.py](../models/MDOF_LNN.py)
- **Performance Metrics:**  
  - MSE
- **Dependencies:**  
  - `jax`, `numpy`, `scipy`, `diffrax`
- **Usage Example:**
  - See `./models/model_data_ext.ipynb`
