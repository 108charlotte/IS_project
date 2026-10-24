import torch

def test_rnn(rnn, computer_decisions, player_decisions): 
    rnn.eval()

from collections import Counter

# function written by ChatGPT
def get_majority_baseline(input_data): 
    labels = [
        int(input_data[p][d]) - 1
        for p in range(len(input_data))
        for d in range(len(input_data[p]))
    ]

    counts = Counter(labels)

    print("Class counts:", counts)
    print(
        "Majority-class baseline:",
        f"{max(counts.values()) / len(labels):.6f}"
    )

# also written by ChatGPT
def get_avg_baseline(train_targets, val_targets):
    train_targets = torch.as_tensor(train_targets, dtype=torch.float32)
    val_targets = torch.as_tensor(val_targets, dtype=torch.float32)

    baseline_pred = train_targets.mean()
    baseline_mse = ((val_targets - baseline_pred) ** 2).mean()

    print("Baseline validation MSE:", baseline_mse.item())
    return baseline_mse.item()