import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import resize
from numpy import linalg

def show_test_faces(folder_path, img_size=(120, 80)):
    #Mostrando imagens de teste em escala de cinza
    files = os.listdir(folder_path)
    num_files = len([f for f in files if os.path.isfile(os.path.join(folder_path, f))])
    fig, axes = plt.subplots(num_files, 2, figsize=(15, 12))
    for index, filename in enumerate(files):
        img_path = os.path.join(folder_path, filename)
        image = img.imread(img_path)
        image_manipulated = rgb2gray(image)
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

    plt.tight_layout()
    plt.show()
    
    
def build_matrix(folder, img_size=(120, 80)):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        image = img.imread(img_path)
        image_manipulated = rgb2gray(image)
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
            if not np.isclose(dot_product, 0):
                ortogonal = False
                break
        if not ortogonal:
            break
    if ortogonal:
        print("Os vetores são ortogonais.")
    else:
        print("Os vetores não são ortogonais.")


def build_matrix_eigenfaces(folder, img_size=(120, 80), use_mean_face=True):
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
    

def visualize_best_eigenface(matrix_eigenfaces, img_size=(120, 80)):
    max_eigenvector = matrix_eigenfaces[:, 0]   #Autovetor associado ao maior autovalor
    eigenface = max_eigenvector.reshape(img_size)    #Reconstruindo a imagem a partir do autovetor obtido
    plt.figure(figsize=(6, 8))
    plt.imshow(eigenface, cmap='gray')
    plt.title('Eigenface associada ao maior autovalor')
    plt.axis('off')
    plt.show()


def approximate_image(path_image, matrix_eigenfaces, mean_face, num_eigenfaces, img_size=(120, 80)):
    #Tentando aproximar uma imagem que não estava no conjunto de treinamento:
    new_image = img.imread(path_image)
    new_image_manipulated = rgb2gray(new_image)
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


img_size = (200, 150)
#show_test_faces('face_images_test', img_size)
matrix_eigenfaces, mean_face = build_matrix_eigenfaces('face_images_test', img_size, False)
#visualize_best_eigenface(matrix_eigenfaces, img_size)
approximate_image('img_teste.jpg', matrix_eigenfaces, mean_face, 8, img_size)