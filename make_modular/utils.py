import matplotlib.pyplot as plt
import numpy as np
import torch



# show some images from dataset with dataLoader
def show_images(images, titles=None, num_images_toShow=8, cols=4, figsize=(7, 4)):
    if isinstance(titles, torch.Tensor):
        titles = titles.numpy()

    converted_images = []
    for image in images:
        if isinstance(image, torch.Tensor):
            converted_images.append(image.numpy().transpose(1, 2, 0))
        else:
            converted_images.append(image)
    
    num_images = num_images_toShow #len(converted_images)
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
