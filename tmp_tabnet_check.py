import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from pytorch_tabnet.tab_model import TabNetClassifier

train_df = pd.read_csv(r'C:\Users\User\ODL_Labs\Assignment\data\train_churn_dataset.csv')
val_df = pd.read_csv(r'C:\Users\User\ODL_Labs\Assignment\data\val_churn_dataset.csv')
test_df = pd.read_csv(r'C:\Users\User\ODL_Labs\Assignment\data\test_churn_dataset.csv')
TARGET='Churn'
features = [c for c in train_df.columns if c != TARGET]
X_train = train_df[features].astype(np.float32).values
y_train = train_df[TARGET].astype(np.int64).values
X_val = val_df[features].astype(np.float32).values
y_val = val_df[TARGET].astype(np.int64).values
X_test = test_df[features].astype(np.float32).values
y_test = test_df[TARGET].astype(np.int64).values
model = TabNetClassifier(
    optimizer_fn=torch.optim.Adam,
    optimizer_params={'lr': 2e-2, 'weight_decay': 1e-4},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    scheduler_params={'step_size': 10, 'gamma': 0.9},
    mask_type='entmax',
    n_d=32, n_a=32, n_steps=5, gamma=1.3, lambda_sparse=1e-3,
    seed=42, verbose=0, device_name='cpu'
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric=['auc','balanced_accuracy'], max_epochs=3, patience=5, batch_size=2048, virtual_batch_size=128, num_workers=0, drop_last=False)
prob = model.predict_proba(X_test)[:, 1]
print('test_auc', round(float(roc_auc_score(y_test, prob)), 4))
print('best_epoch', model.best_epoch_)
