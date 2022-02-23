import matplotlib.pyplot as plt


def plot(loss_history, train_acc_history, val_acc_history):
    plt.rcParams['figure.figsize'] = (16.0, 16.0)
    plt.subplot(2, 1, 1)
    plt.plot(loss_history)
    plt.title('Loss history')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(train_acc_history, label='train')
    plt.plot(val_acc_history, label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Classification accuracy')
    plt.legend()

    plt.show()


def save(loss_history, train_acc_history, val_acc_history, fname):
    plt.rcParams['figure.figsize'] = (16.0, 16.0)
    plt.subplot(2, 1, 1)
    plt.plot(loss_history)
    plt.title('Loss history')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(train_acc_history, label='train')
    plt.plot(val_acc_history, label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Classification accuracy')
    plt.legend()

    plt.savefig(fname, bbox_inches='tight')
    plt.close()
