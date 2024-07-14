import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import resize
from numpy import linalg

img_size = (200, 150)

def show_test_faces(folder_path, limit=5):
    #Mostrando imagens de teste em escala de cinza
    files = os.listdir(folder_path)
    num_files = len([f for f in files if os.path.isfile(os.path.join(folder_path, f))])
    print(f"Numfiles: {num_files}")
    if limit < num_files:
        fig, axes = plt.subplots(limit, 2, figsize=(15, 12))
        for index, filename in enumerate(files):
            if index < limit:
                img_path = os.path.join(folder_path, filename)
                image = img.imread(img_path)
                image_manipulated = image
                if image_manipulated.ndim == 3:
                    image_manipulated = rgb2gray(image_manipulated)
                image_manipulated = resize(image_manipulated, img_size, anti_aliasing=True)

                if num_files > 1:
                    axes[index, 0].imshow(image, cmap='gray')
                    axes[index, 0].set_title(f'Pessoa {index+1}')
                    axes[index, 0].axis('off')
                    axes[index, 1].imshow(image_manipulated, cmap='gray')
                    axes[index, 1].set_title(f'Pessoa {index+1} reduzida')
                    axes[index, 1].axis('off')
                else:
                    axes[0].imshow(image, cmap='gray')
                    axes[0].set_title(f'Pessoa {index+1}')
                    axes[0].axis('off')
                    axes[1].imshow(image_manipulated, cmap='gray')
                    axes[1].set_title(f'Pessoa {index+1} reduzida')
                    axes[1].axis('off')
            else:
                break

        #plt.tight_layout()
        plt.show()
    
    
def build_matrix(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        image = img.imread(img_path)
        image_manipulated = image
        if image_manipulated.ndim == 3:
            image_manipulated = rgb2gray(image_manipulated)
        image_manipulated = resize(image_manipulated, img_size, anti_aliasing=True)
        images.append(image_manipulated.flatten())    #Convertendo as imagens para vetores
    
    images_matrix = np.array(images).T    #Matriz onde cada coluna é uma das imagens
    #images_matrix = np.array(images)    #Matriz onde cada linha é uma das imagens
    return images_matrix


def isOrtogonal(matrix):
    num_colunas = matrix.shape[1]
    ortogonal = True
    for i in range(num_colunas):
        for j in range(i + 1, num_colunas):
            dot_product = np.dot(matrix[:, i], matrix[:, j])
            if not np.isclose(dot_product, 0, atol=0.0):
                ortogonal = False
                break
        if not ortogonal:
            break
    if ortogonal:
        print("Os vetores são ortogonais.")
    else:
        print("Os vetores não são ortogonais.")


def build_matrix_eigenfaces(folder, use_mean_face=True):
    images_matrix = build_matrix(folder, img_size)
    #print(f"Dimensões da matriz de imagens: {images_matrix.shape}")
    mean_face = np.zeros(images_matrix.shape[0])
    if use_mean_face:
        for i in range(images_matrix.shape[1]):
            mean_face += images_matrix[:, i]
        mean_face /= images_matrix.shape[1]
        images_matrix = images_matrix - mean_face[:, np.newaxis]   #Subtraindo o vetor médio de cada imagem
    
    simple_covariance_matrix = np.matmul(images_matrix.T, images_matrix)
    eigenvalues, eigenvectors = linalg.eig(simple_covariance_matrix)   #Autovalores e autovetores da matriz mais simples (os autovetores são as colunas de uma matriz)
    #print(f"Autovalores: {eigenvalues}\nAutovetores:\n{eigenvectors}")

    eigen_pairs = [(eigenvalues[i], eigenvectors[:, i]) for i in range(len(eigenvalues))]
    eigen_pairs.sort(key=lambda x: x[0], reverse=True)   #Ordenando os pares de autovetores e autovalores em ordem decrescente em relação aos autovalores
    #print(f"\nPares:\n{eigen_pairs}")
    
    num_eigenfaces = len(eigen_pairs)  #Número de eigenfaces principais a serem usadas
    eigenvectors_matrix = np.array([pair[1] for pair in eigen_pairs[:num_eigenfaces]]).T  #Matriz onde cada coluna é um autovetor da matriz de covariância simplificada
    #print(eigenvectors_matrix)
    eigenfaces_matrix = np.matmul(images_matrix, eigenvectors_matrix)    #Matriz onde cada coluna é um autovetor da matriz de covariância original
    
    for i in range(eigenfaces_matrix.shape[1]):    #Normalizando os autovetores
        coluna = eigenfaces_matrix[:, i]
        #print(f"Norma da coluna: {np.linalg.norm(coluna)}")
        eigenfaces_matrix[:, i] /= linalg.norm(coluna)
    
    #isOrtogonal(eigenfaces_matrix)
    return eigenfaces_matrix, mean_face
    

def visualize_eigenfaces(matrix_eigenfaces):
    limit = min(40, matrix_eigenfaces.shape[1])   #Mostrando as primeiras 40 eigenfaces
    eigenfaces = []
    for i in range(limit):
        eigenvector = matrix_eigenfaces[:, i]   #Autovetor
        eigenface = eigenvector.reshape(img_size)    #Reconstruindo a imagem a partir do autovetor obtido
        eigenfaces.append(eigenface)
    
    num_eigenfaces = len(eigenfaces)
    fig, axes = plt.subplots(5, 8, figsize=(15, 12))
    for i, ax in enumerate(axes.flat):
        if i < num_eigenfaces:
            ax.imshow(eigenfaces[i], cmap='gray')
            ax.set_title(f'Eigenface {i+1}')
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()


def approximate_image(path_image, matrix_eigenfaces, mean_face, num_eigenfaces):
    #Tentando aproximar uma imagem que não estava no conjunto de treinamento:
    new_image = img.imread(path_image)
    new_image_manipulated = new_image
    if new_image_manipulated.ndim == 3:
        new_image_manipulated = rgb2gray(new_image_manipulated)
    new_image_manipulated = resize(new_image_manipulated, img_size, anti_aliasing=True)
    new_image_vector = (new_image_manipulated.flatten() - mean_face)
    
    work_matrix = matrix_eigenfaces[:, :num_eigenfaces]
    
    weights = np.matmul(work_matrix.T, new_image_vector)  #Matriz de tamanhos das projeções da imagem no subespaço gerado pelos autovetores
    reconstructed_image_vector = np.matmul(work_matrix, weights)
    reconstructed_image_vector = mean_face + reconstructed_image_vector
    reconstructed_image = reconstructed_image_vector.reshape(new_image_manipulated.shape)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 12))
    axes[0].imshow(new_image_manipulated, cmap='gray')
    axes[0].set_title(f'Original')
    axes[0].axis('off')
    axes[1].imshow(reconstructed_image, cmap='gray')
    axes[1].set_title(f'Imagem Reconstruída')
    axes[1].axis('off')
    plt.show()


def images_classification(path_image1, path_image2, matrix_eigenfaces, mean_face):
    #Classificando duas imagens como sendo da mesma pessoa ou de pessoas diferentes plotando em duas dimensões:
    image1 = img.imread(path_image1)
    image1_manipulated = image1
    if image1_manipulated.ndim == 3:
        image1_manipulated = rgb2gray(image1_manipulated)
    image1_manipulated = resize(image1_manipulated, img_size, anti_aliasing=True)
    image1_vector = (image1_manipulated.flatten() - mean_face)
    
    image2 = img.imread(path_image2)
    image2_manipulated = image2
    if image2_manipulated.ndim == 3:
        image2_manipulated = rgb2gray(image2_manipulated)
    image2_manipulated = resize(image1_manipulated, img_size, anti_aliasing=True)
    image2_vector = (image2_manipulated.flatten() - mean_face)
    
    #TODO



#show_test_faces('att_faces_png')
matrix_eigenfaces, mean_face = build_matrix_eigenfaces('att_faces_png', True)
#print(matrix_eigenfaces.shape)
visualize_eigenfaces(matrix_eigenfaces)
#approximate_image('1_1.png', matrix_eigenfaces, mean_face, matrix_eigenfaces.shape[1])