import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# show some images from dataset with dataLoader
def show_images(images, titles=None, num_images_to_show=8, cols=4, figsize=(7, 4)):
    if isinstance(titles, torch.Tensor):
        titles = titles.numpy()

    converted_images = []
    for image in images:
        if isinstance(image, torch.Tensor):
            converted_images.append(image.numpy().transpose(1, 2, 0))
        else:
            converted_images.append(image)
    
    num_images = num_images_to_show #len(converted_images)
    rows = np.ceil(num_images / cols).astype(int)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(converted_images[i], cmap='gray')
            ax.axis('off')
            ax.set_title(titles[i] if titles is not None else '')
        else:
            ax.axis('off')
            
    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    plt.tight_layout()
    plt.show()



# calculate f1_score and recall and precision
def calculate_f1Score_recall_precision(preds, labels, num_classes):
    # lists for save all recall and precision for all classes
    recallForEachClass = []
    precisionForEachClass = []

    # our classes are from 0 to 9
    for i in range(num_classes):
        # create list of True False and then sum Trues in list
        TP = np.sum([x == i and y == i for x, y in zip(labels, preds)])
        FN = np.sum([x == i and y != i for x, y in zip(labels, preds)])
        FP = np.sum([x != i and y == i for x, y in zip(labels, preds)])

        # recall and precision for each class
        if (TP + FN) == 0 :
            recall = 0
        else:
            recall = TP / (TP + FN)

        if (TP + FP) == 0 : 
            precision = 0
        else:
            precision = TP / (TP + FP)
        

        recallForEachClass.append(recall)
        precisionForEachClass.append(precision)

    # average recall,precision for all classes
    avg_recall = np.sum(recallForEachClass) / num_classes
    avg_precision = np.sum(precisionForEachClass) / num_classes

    # f1_score
    if (avg_recall + avg_precision) == 0 : 
        f1_score = 0
    else:
        f1_score = (2 * avg_recall * avg_precision) / (avg_recall + avg_precision)

    return f1_score, avg_recall, avg_precision



# create confusion matrix with model predicted and labels
def plot_confusion_matrix(y_true :np.array, y_pred: np.array, labels: np.array):
    '''
    y_true : real labels
    y_pred : models prediction
    labels : all possible labels exp --> [0, 1, ... 9]

    '''
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    dis = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    dis.plot()



# plot loss-acc-f1score curves
def plot_loss_curves(results, figsize=(12, 3)):
    # Define the figure size (width, height)
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # loss curves
    epochs = range(len(results['train_loss']))
    axes[0].plot(epochs, results['train_loss'], label='train_loss')
    axes[0].plot(epochs, results['val_loss'], label='val_loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend(loc='best')

    # acc curves
    axes[1].plot(epochs, results['train_acc'], label='train_acc')
    axes[1].plot(epochs, results['val_acc'], label='val_acc')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend(loc='best')

    # f1_score curves
    axes[2].plot(epochs, results['train_f1_score'], label='train_f1_score')
    axes[2].plot(epochs, results['val_f1_score'], label='val_f1_score')
    axes[2].set_title(f'F1_score')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('F1_score')
    axes[2].legend(loc='best')

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.3)



# find best lr for each model
from make_modular.engine import train
from make_modular.configs import device

def find_best_lr(model_class, n_iter, epochs_per_lr, lr_low, lr_high, train_dl, val_dl):
    ''' n_iter: number of iteratins we want to find best lr with different lrs.
        epochs_per_lr: number of epochs for one lr .
        interval for lr : (lr_low, lr_high) 
    number of totall runs(epochs) is : n_iter * epochs_per_lr
    '''
    loss_fn = nn.CrossEntropyLoss()
    final_res_lr = []
    for _ in range(n_iter):

        model = model_class()
        lr = 10 ** (np.random.uniform(low=lr_low, high=lr_high))
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        print(f'lr : {lr}')

        results = train(model=model,
                        train_dl=train_dl,
                        val_dl=val_dl,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        device=device,
                        epochs=epochs_per_lr)
        final_res_lr.append({lr:results})
        print('---' * 40)

    return final_res_lr



# the train time 
def print_train_time(start, end):
    total_time = end - start
    print(f"\nTrain time: {total_time/60:.3f} minutes")
    return total_time
