import json
import matplotlib.pyplot as plt

def display_results(out, report_every=None, charts=True, accuracy=False):
    output = json.dumps(out, indent=4, sort_keys=True)
    if not charts: 
        print(output)
    else: # report_every is always included if I want to show the charts
        print(f"Average loss improvement: {out['avg_loss_improvement']}\nAverage test loss improvement: {out['avg_test_loss_improvement']}")
        # dynamic for different outputs
        if not accuracy: 
            fig, axes = plt.subplots(1,2,figsize=(12,8))
        else: 
            fig, axes = plt.subplots(1, 3, figsize=(12,8))

        train_x = range(1, len(out['losses'])+1)
        test_x = range(1, len(out['test_losses'])+1)
        test_x = [item * report_every for item in test_x] # each element multiplied by report_every to get the epoch it actually occured in

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

        if accuracy: # test accuracy, same x as test_x
            axes[2].plot(test_x, out['accuracies'], marker='o')
            axes[2].set_title("Test Accuracy")
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("Accuracy")
            axes[2].grid(True)

        plt.show()