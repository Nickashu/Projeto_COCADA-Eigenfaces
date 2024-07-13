import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import resize
from numpy import linalg


def load_images(folder, img_size=(120, 80)):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        image = img.imread(img_path)
        if image.ndim == 3:   #Se a imagem for colorida, quero transformá-la para uma imagem em escala de cinza
            image = rgb2gray(image)
        image_resized = resize(image, img_size, anti_aliasing=True)
        images.append(image_resized.flatten())    #Convertendo as imagens para vetores
    return np.array(images)



def show_test_faces(folder_path, img_size=(120, 80)):
    files = os.listdir(folder_path)
    num_files = len([f for f in files if os.path.isfile(os.path.join(folder_path, f))])
    fig, axes = plt.subplots(num_files, 2, figsize=(15, 12))
    
    for index, filename in enumerate(files):
        img_path = os.path.join(folder_path, filename)
        image = img.imread(img_path)
        image_manipulated = rgb2gray(image)
        image_manipulated = resize(image_manipulated, img_size, anti_aliasing=True)
        #images.append(image_resized.flatten())    #Convertendo as imagens para vetores
        
        axes[index, 0].imshow(image, cmap='gray')
        axes[index, 0].set_title(f'Pessoa {index+1}')
        axes[index, 0].axis('off')
        axes[index, 1].imshow(image_manipulated, cmap='gray')
        axes[index, 1].set_title(f'Pessoa {index+1} reduzida')
        axes[index, 1].axis('off')

    plt.tight_layout()
    plt.show()


show_test_faces('face_images_test')