# Heart Failure Prediction with PyTorch
 
A learning project focused on understanding and implementing core **PyTorch concepts** for binary classification. The goal wasn't to build a perfect model — it was to get hands-on with how PyTorch actually works.
 
---
 
## What I was learning
 
- Building a custom `Dataset` and `DataLoader`
- Defining a neural network using `nn.Module`
- Writing a training loop from scratch (forward pass, loss, backprop, optimizer step)
- Using `BCELoss` and `Adam` optimizer
- Understanding what **Dropout** does and comparing training with and without it
- Evaluating a model on a test set
---
 
## Dataset
 
**[Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)** by fedesoriano on Kaggle.
 
The dataset was just a vehicle to practice PyTorch — EDA wasn't the focus here so it's minimal (basic `.head()`, `.info()`, class distribution).
 
---
 
## Model Architecture
 
```
Input (11 features)
    → Linear(11 → 32) → ReLU → [Dropout(0.3)]
    → Linear(32 → 16) → ReLU → [Dropout(0.3)]
    → Linear(16 → 1)  → Sigmoid
```
 
Two versions trained side-by-side — one **with Dropout** and one **without** — to see the difference in loss curves and test accuracy.
 
---
 
## Run it
 
Open on Kaggle, add the [Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) dataset to the notebook and run all cells.
 
---
 
## Author
 
**Rugved** — [Kaggle](https://www.kaggle.com/rugvedbane)
 
