import torch

# function written by ChatGPT - what you would get if you only predicted the majority label (useful for emotion nn, not for personality nn bc output for pnn is continuous)
def get_majority_baseline(train_targets, val_targets):
    train_targets = torch.as_tensor(train_targets, dtype=torch.float32)
    val_targets = torch.as_tensor(val_targets, dtype=torch.float32)

    # Find the most common class in the training set
    majority_class = torch.mode(train_targets).values

    # Predict that class for every validation example
    predictions = torch.full_like(val_targets, majority_class.item())

    # Calculate validation MSE
    mse = torch.mean((val_targets - predictions) ** 2)

    return float(mse.item())

# also written by ChatGPT - basically, if you use the average of the training targets to predict validation labels, this is what you would get
def get_avg_baseline(train_targets, val_targets):
    train_targets = torch.as_tensor(train_targets, dtype=torch.float32)-1
    val_targets = torch.as_tensor(val_targets, dtype=torch.float32)-1

    baseline_pred = train_targets.mean()
    baseline_mse = ((val_targets - baseline_pred) ** 2).mean()

    # print("Baseline validation MSE:", baseline_mse.item())
    return float(baseline_mse.item())