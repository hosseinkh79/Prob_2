import matplotlib.pyplot as plt
import numpy as np
import torch



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



