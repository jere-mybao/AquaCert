# AquaCert: MLOps-Driven Water Potability Prediction

AquaCert demonstrates a comprehensive implementation of MLOps principles to solve a real-world problem: predicting water potability using machine learning. This project integrates state-of-the-art tools like **MLflow** for experiment tracking, **DVC** for data and model versioning, and a **Tkinter**-based desktop application for deployment, ensuring both technical rigor and practical usability.

---

## Project Architecture

AquaCert is structured around a robust MLOps pipeline, emphasizing reproducibility, scalability, and seamless deployment.

### Key Components:
1. **Data Management**:
   - **Dataset**: [Water Potability Dataset](https://raw.githubusercontent.com/Sarthak-1408/Water-Potability/refs/heads/main/water_potability.csv) containing water quality metrics.
   - **Versioning**: Handled via **DVC**, ensuring consistent dataset tracking.
   - **Preprocessing**: Techniques include handling missing data using mean and median imputations.

2. **Model Development**:
   - Multiple experiments conducted with:
     - **Random Forest** (Best-performing model)
     - **Logistic Regression**
     - **XGBoost**
   - **Optimization**: Hyperparameter tuning to identify `n_estimators=1000`, `max_depth=None` as optimal for Random Forest.
   - **Evaluation**: Performance logged using **MLflow** for metrics, parameters, and artifacts.

3. **Model Registration**:
   - Best model registered in the **MLflow Model Registry** for streamlined integration and deployment.

4. **Deployment**:
   - **Tkinter Desktop Application**: A lightweight GUI fetches the latest model from MLflow, enabling real-time water potability predictions.

---

## Repository Structure

- `data/`: Raw and processed datasets managed via DVC.
- `models/`: Trained models and evaluation metrics.
- `notebooks/`: Experimental data analysis and model prototyping.
- `src/`: Core scripts for pipeline stages including data preprocessing, training, and evaluation.
- `GUI.py`: Tkinter-based desktop application for predictions.

---

## Getting Started

### 1. Clone the Repository
```bash
git clone git@github.com:jere-mybao/aquacert.git
