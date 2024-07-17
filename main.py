import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
from skimage.color import rgb2gray
from skimage.transform import resize
from numpy import linalg

img_size = (240, 160)
list_vector_images = []

yale_database_path = "yale_database"
att_database_path = "att_database"
yale_att_database_path = "yale_att_database"

image_approximation1_path = "teste_aproximacao1.png"
image_approximation2_path = "teste_aproximacao2.png"
image_approximation3_path = "teste_aproximacao3.jpeg"
image_recognition1_path = "teste_reconhecimento1.png"
image_recognition2_path = "teste_reconhecimento2.png"
person1_folder_path = "person1"
person2_folder_path = "person2"

def show_test_faces(folder_path, limit=5):
    #Mostrando imagens de teste em escala de cinza
    files = os.listdir(folder_path)
    num_files = len([f for f in files if os.path.isfile(os.path.join(folder_path, f))])
    #print(f"Numfiles: {num_files}")
    if limit < num_files:
        fig, axes = plt.subplots(limit, 2, figsize=(12, 10))
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
                    axes[index, 0].set_title(f'Imagem original')
                    axes[index, 0].axis('off')
                    axes[index, 1].imshow(image_manipulated, cmap='gray')
                    axes[index, 1].set_title(f'Imagem manipulada')
                    axes[index, 1].axis('off')
                else:
                    axes[0].imshow(image, cmap='gray')
                    axes[0].set_title(f'Imagem original')
                    axes[0].axis('off')
                    axes[1].imshow(image_manipulated, cmap='gray')
                    axes[1].set_title(f'Imagem manipulada')
                    axes[1].axis('off')
            else:
                break

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
        vector_image = image_manipulated.flatten()
        images.append(vector_image)    #Convertendo as imagens para vetores
        list_vector_images.append(vector_image)
    
    images_matrix = np.array(images).T    #Matriz onde cada coluna é uma das imagens
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

def show_mean_face(mean_face):
    reconstructed_image = mean_face.reshape(img_size)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title('Mean Face')
    plt.axis('off')
    plt.show()


def build_matrix_eigenfaces(folder, use_mean_face=True):
    images_matrix = build_matrix(folder)
    mean_face = np.zeros(images_matrix.shape[0])
    if use_mean_face:
        for i in range(images_matrix.shape[1]):
            mean_face += images_matrix[:, i]
        mean_face /= images_matrix.shape[1]
        images_matrix = images_matrix - mean_face[:, np.newaxis]   #Subtraindo o vetor médio de cada imagem
    
    simple_covariance_matrix = np.matmul(images_matrix.T, images_matrix)
    eigenvalues, eigenvectors = linalg.eig(simple_covariance_matrix)   #Autovalores e autovetores da matriz mais simples (os autovetores são as colunas de uma matriz)

    eigen_pairs = [(eigenvalues[i], eigenvectors[:, i]) for i in range(len(eigenvalues))]
    eigen_pairs.sort(key=lambda x: x[0], reverse=True)   #Ordenando os pares de autovetores e autovalores em ordem decrescente em relação aos autovalores
    
    num_eigenfaces = len(eigen_pairs)  #Número de eigenfaces principais a serem usadas
    eigenvectors_matrix = np.array([pair[1] for pair in eigen_pairs[:num_eigenfaces]]).T  #Matriz onde cada coluna é um autovetor da matriz de covariância simplificada
    eigenfaces_matrix = np.matmul(images_matrix, eigenvectors_matrix)    #Matriz onde cada coluna é um autovetor da matriz de covariância original
    
    for i in range(eigenfaces_matrix.shape[1]):    #Normalizando os autovetores
        coluna = eigenfaces_matrix[:, i]
        eigenfaces_matrix[:, i] /= linalg.norm(coluna)
        
    return eigenfaces_matrix, mean_face

def visualize_eigenfaces(matrix_eigenfaces):
    limit = min(40, matrix_eigenfaces.shape[1])   #Mostrando as primeiras 40 eigenfaces
    eigenfaces = []
    for i in range(limit):
        eigenvector = matrix_eigenfaces[:, i]   #Autovetor
        eigenface = eigenvector.reshape(img_size)    #Reconstruindo a imagem a partir do autovetor obtido
        eigenfaces.append(eigenface)
    
    num_eigenfaces = len(eigenfaces)
    fig, axes = plt.subplots(5, 8, figsize=(10, 7))
    for i, ax in enumerate(axes.flat):
        if i < num_eigenfaces:
            ax.imshow(eigenfaces[i], cmap='gray')
            ax.set_title(f'Eigenface {i+1}')
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()

def approximate_image(path_image, matrix_eigenfaces, mean_face):
    #Tentando aproximar uma imagem que não estava no conjunto de treinamento:
    new_image = img.imread(path_image)
    new_image_manipulated = new_image
    if new_image_manipulated.ndim == 3:
        new_image_manipulated = rgb2gray(new_image_manipulated)
    new_image_manipulated = resize(new_image_manipulated, img_size, anti_aliasing=True)
    new_image_vector = (new_image_manipulated.flatten() - mean_face)
    
    num_eigenfaces = [1, 10, 50, 100, 400, 800, 1600, 2000]
    num_eigenfaces.append(matrix_eigenfaces.shape[1])

    reconstructed_images = []
    for index, num_eigenface in enumerate(num_eigenfaces):
        work_matrix = matrix_eigenfaces[:, :num_eigenface]
        weights = np.matmul(work_matrix.T, new_image_vector)  #Vetor com os tamanhos das projeções da imagem no subespaço gerado pelos autovetores
        reconstructed_image_vector = np.matmul(work_matrix, weights)
        reconstructed_image_vector = mean_face + reconstructed_image_vector
        reconstructed_image = reconstructed_image_vector.reshape(img_size)
        reconstructed_images.append(reconstructed_image)
    
    fig, axes = plt.subplots(3, 4, figsize=(10, 7))
    axes[0, 0].imshow(new_image_manipulated, cmap='gray')
    axes[0, 0].set_title(f'Original')
    axes[0, 0].axis('off')
    for i, ax in enumerate(axes.flat[1:], start=1):
        if i-1 < len(reconstructed_images):
            ax.imshow(reconstructed_images[i-1], cmap='gray')
            ax.set_title(f'{num_eigenfaces[i-1]} eigenfaces')
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()

def images_classification(path_person1, path_person2, matrix_eigenfaces, mean_face):
    #Comparando dua imagens em um espaço bidimensional:
    work_matrix = matrix_eigenfaces[:, [4, 5]]    #Quero projetar no plano formado pelas eigenfaces 5 e 6
    person1 = []
    person2 = []
    for filename in os.listdir(path_person1):
        img_path = os.path.join(path_person1, filename)
        image = img.imread(img_path)
        image_manipulated = image
        if image_manipulated.ndim == 3:
            image_manipulated = rgb2gray(image_manipulated)
        image_manipulated = resize(image_manipulated, img_size, anti_aliasing=True)
        image_vector = (image_manipulated.flatten() - mean_face)
        weights_image = np.matmul(work_matrix.T, image_vector)
        person1.append(weights_image)
    for filename in os.listdir(path_person2):
        img_path = os.path.join(path_person2, filename)
        image = img.imread(img_path)
        image_manipulated = image
        if image_manipulated.ndim == 3:
            image_manipulated = rgb2gray(image_manipulated)
        image_manipulated = resize(image_manipulated, img_size, anti_aliasing=True)
        image_vector = (image_manipulated.flatten() - mean_face)
        weights_image = np.matmul(work_matrix.T, image_vector)
        person2.append(weights_image)

    # Extraindo as coordenadas x e y dos vetores
    x1, y1 = zip(*person1)
    x2, y2 = zip(*person2)
    
    plt.scatter(x1, y1, color='blue', marker='o', label='Pessoa 1')
    plt.scatter(x2, y2, color='red', marker='x', label='Pessoa 2')
    plt.title('Gráfico dos Pontos Representados por Duas Pessoas')
    plt.xlabel('Eigenface 5')
    plt.ylabel('Eigenface 6')
    plt.grid(True)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()


def person_recognition(path_image, matrix_eigenfaces, mean_face, num_eigenfaces):
    #Dada uma imagem que não está no conjunto de treinamento, quero achar a melhor aproximação para ela (usando distâncias euclidianas)
    image = img.imread(path_image)
    image_manipulated = image
    if image_manipulated.ndim == 3:
        image_manipulated = rgb2gray(image_manipulated)
    image_manipulated = resize(image_manipulated, img_size, anti_aliasing=True)
    image_vector = (image_manipulated.flatten() - mean_face)

    work_matrix = matrix_eigenfaces[:, :num_eigenfaces]    #Quanto maior o número de eigenfaces, mais preciso e demorado será
    
    weights = np.matmul(work_matrix.T, image_vector)
    best_vector_image = list_vector_images[0]
    min_distance = 0
    
    for index, vec_image in enumerate(list_vector_images):
        vec_image_manipulated = vec_image - mean_face
        weights_compare = np.matmul(work_matrix.T, vec_image_manipulated)
        euclidian_distance = linalg.norm(weights - weights_compare)    #Calculando a distância Euclidiana entre os dois vetores de tamanhos
        if index == 0:
            best_vector_image = vec_image
            min_distance = euclidian_distance
        else:
            if euclidian_distance < min_distance:
                best_vector_image = vec_image
                min_distance = euclidian_distance
    
    best_image = best_vector_image.reshape(img_size)
    fig, axes = plt.subplots(1, 2, figsize=(10, 7))
    axes[0].imshow(image_manipulated, cmap='gray')
    axes[0].set_title(f'Pessoa')
    axes[0].axis('off')
    axes[1].imshow(best_image, cmap='gray')
    axes[1].set_title(f'Melhor aproximação')
    axes[1].axis('off')
    plt.show()
    

matrix_eigenfaces, mean_face = build_matrix_eigenfaces(yale_att_database_path)   #Montando a matriz de autofaces
visualize_eigenfaces(matrix_eigenfaces)                                          #Visualizando as autofaces (eigenfaces)
#approximate_image(image_approximation1_path, matrix_eigenfaces, mean_face)       #Exemplo de aproximação de um rosto
#images_classification(person1_folder_path, person2_folder_path, matrix_eigenfaces, mean_face)   #Exemplo de classificação de imagens de duas pessoas
#person_recognition(image_recognition1_path, matrix_eigenfaces, mean_face, 100)   #Exemplo de reconhecimento facial usando 100 autovetores