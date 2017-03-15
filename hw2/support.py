import matplotlib.pyplot as plt
from glob import glob
import os

def accuracy_loss_curve(train_acc, test_acc, train_loss, test_loss, iter_steps):
    plt.subplot(2, 1, 1)
    plt.plot(iter_steps, train_loss, '-o', label ='train')
    plt.plot(iter_steps, test_loss, '-o', label = 'test')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.subplot(2, 1, 2)
    plt.plot(iter_steps, train_acc, '-o', label='train')
    plt.plot(iter_steps, test_acc, '-o', label='test')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.gcf().set_size_inches(15, 12)
    plt.show()


def get_run_var(dir):
    subdirectories = get_immediate_subdirectories(dir)
    return len(subdirectories)



def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

