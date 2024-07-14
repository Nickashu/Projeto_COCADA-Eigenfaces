import os
from PIL import Image

# Caminho da pasta contendo as imagens PGM
input_folder = 'att_faces'
output_folder = 'att_faces_png'

# Percorrer todos os arquivos na pasta de entrada
for index, filename in enumerate(os.listdir(input_folder)):
    folder_path = os.path.join(input_folder, filename)
    print(folder_path)
    for filenamePGM in os.listdir(folder_path):
        # Caminho completo para o arquivo PGM
        pgm_path = os.path.join(folder_path, filenamePGM)
        # Caminho completo para o arquivo PNG de saída
        png_filename = f'{index+1}_' + os.path.splitext(filenamePGM)[0] + '.png'
        png_path = os.path.join(output_folder, png_filename)
        
        # Abrir a imagem PGM e convertê-la para PNG
        with Image.open(pgm_path) as img:
            img.save(png_path)

        print(f'Convertido {filenamePGM} para {png_filename}')

print('Conversão concluída!')