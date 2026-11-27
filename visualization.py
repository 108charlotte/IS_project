import json
import matplotlib.pyplot as plt

def display_results(out, report_every=None, charts=True, accuracy=False, multiple=False):
    output = json.dumps(out, indent=4, sort_keys=True)
    if not charts: 
        print(output)
    else: # report_every is always included if I want to show the charts
        if multiple: 
            num_rows = len(out)
            if not accuracy: 
                fig, axes = plt.subplots(num_rows,2,figsize=(12,8))
            else: 
                fig, axes = plt.subplots(num_rows, 3, figsize=(12,8))
            for i, o in enumerate(out): 
                train_x = range(1, len(o['losses'])+1)
                test_x = range(1, len(o['test_losses'])+1)
                test_x = [item * report_every for item in test_x] # each element multiplied by report_every to get the epoch it actually occured in

                # train plot
                axes[i, 0].plot(train_x, o['losses'], marker='o')
                axes[i, 0].set_title("Train Losses")
                axes[i, 0].set_xlabel("Epoch")
                axes[i, 0].set_ylabel("Loss")
                axes[i, 0].grid(True)

                axes[i, 1].plot(test_x, o['test_losses'], marker='o')
                axes[i, 1].set_title("Validation Losses")
                axes[i, 1].set_xlabel("Epoch")
                axes[i, 1].set_ylabel("Loss")
                axes[i, 1].grid(True)

                if accuracy: # test accuracy, same x as test_x
                    axes[i, 2].plot(test_x, o['accuracies'], marker='o')
                    axes[i, 2].set_title("Test Accuracy")
                    axes[i, 2].set_xlabel("Epoch")
                    axes[i, 2].set_ylabel("Accuracy")
                    axes[i, 2].grid(True)

            plt.tight_layout()
            plt.show()
        else: 
            make_chart(out, accuracy, report_every)

def make_chart(out, accuracy, report_every): 
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


def display_pcrnn_results(n_epochs, inference_errors, inference_gm_errors, learning_errors, learning_gm_errors, val_errors, val_accuracies):
    num_runs = len(inference_errors) 
    fig, axes = plt.subplots(num_runs * 3, 2, figsize=(12,8/num_runs), squeeze=False)
    epochs = range(1, n_epochs+1)

    for i in range(num_runs): 
        axes[i*3, 0].plot(epochs, inference_errors[i], marker='o', color='purple')
        axes[i*3, 0].set_title(f"Run {i+1} Error History (Infer)")
        axes[i*3, 0].set_xlabel("Epoch")
        axes[i*3, 0].set_ylabel("Error")
        axes[i*3, 0].grid(True)

        axes[i*3, 1].plot(epochs, inference_gm_errors[i], marker='o', color='purple')
        axes[i*3, 1].set_title(f"Run {i+1} Gain Mod Error History (Infer)")
        axes[i*3, 1].set_xlabel("Epoch")
        axes[i*3, 1].set_ylabel("Error")
        axes[i*3, 1].grid(True)

        axes[i*3 + 1, 0].plot(epochs, learning_errors[i], marker='o', color='green')
        axes[i*3 + 1, 0].set_title(f"Run {i+1} Supervised Readout Error History (Learning)")
        axes[i*3 + 1, 0].set_xlabel("Epoch")
        axes[i*3 + 1, 0].set_ylabel("Error")
        axes[i*3 + 1, 0].grid(True)

        axes[i*3 + 1, 1].plot(epochs, learning_gm_errors[i], marker='o', color='green')
        axes[i*3 + 1, 1].set_title(f"Run {i+1} Supervised Readout Gain Mod Error History (Learning)")
        axes[i*3 + 1, 1].set_xlabel("Epoch")
        axes[i*3 + 1, 1].set_ylabel("Error")
        axes[i*3 + 1, 1].grid(True)

        axes[i*3 + 2, 0].plot(epochs, val_errors[i], marker='o', color='orange')
        axes[i*3 + 2, 0].set_title(f"Run {i+1} Validation Errors")
        axes[i*3 + 2, 0].set_xlabel("Epoch")
        axes[i*3 + 2, 0].set_ylabel("Error")
        axes[i*3 + 2, 0].grid(True)

        axes[i*3 + 2, 1].plot(epochs, val_accuracies[i], marker='o', color='orange')
        axes[i*3 + 2, 1].set_title(f"Run {i+1} Validation Accuracy")
        axes[i*3 + 2, 1].set_xlabel("Epoch")
        axes[i*3 + 2, 1].set_ylabel("Accuracy")
        axes[i*3 + 2, 1].grid(True)

    plt.tight_layout()
    plt.show()