import matplotlib.pyplot as plt

def plot_history(history, save_path=None):
    """Plot training and validation accuracy/loss from Keras history.

    If `save_path` is provided, the figure will be saved to that path instead
    of being shown interactively.
    """
    fig = plt.figure(figsize=(12, 4))

    # Accuracy plot
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Accuracy over epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Loss plot
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Loss over epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def evaluate_model(model, validation_generator):
    """Evaluate model and print validation accuracy."""
    loss, accuracy = model.evaluate(validation_generator)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    return loss, accuracy
