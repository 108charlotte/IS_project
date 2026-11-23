import json
import matplotlib.pyplot as plt

def display_results(out, report_every=None, charts=True):
    output = json.dumps(out, indent=4, sort_keys=True)
    if not charts: 
        print(output)
    else: 
        print(f"Average loss improvement: {out['avg_loss_improvement']}\nAverage test loss improvement: {out['avg_test_loss_improvement']}")
        fig, axes = plt.subplots(1,2,figsize=(12,8))

        train_x = range(1, len(out['losses'])+1)
        test_x = range(1, len(out['test_losses']+1))*report_every # each element multiplied by report_every to get the epoch it actually occured in

        # train plot
        axes[0].plot(train_x, out['losses'], marker='o')
        axes[0].set_title("Train Losses")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True)

        axes[1].plot(test_x, out['test_losses'], marker='o')
        axes[1].set_title("Validation Losses")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].grid(True)

        plt.show()