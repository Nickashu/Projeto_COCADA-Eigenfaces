# Projeto Final COCADA - Eigenfaces

### Contexto Geral
Projeto desenvolvido como trabalho final da disciplina Computação Científica e Análise de Dados (COCADA).

Este projeto buscou se aprofundar em um método que faz uso do PCA e pode servir de base para sistemas de reconhecimento facial, aproximação de rostos e compressão de imagens. O método se utiliza de “eigenfaces” (autofaces, em uma tradução livre). Eigenfaces nada mais são do que autovetores usados para representar uma matriz formada por imagens de rostos de diferentes indivíduos. A técnica baseia-se no PCA, que é usado para reduzir a dimensionalidade dos dados e identificar os padrões mais significativos nas diferentes faces, capturando as variações mais importantes dentro deste conjunto de imagens.

### Funcionalidades Implementadas
Entre as diversas funcionalidades das eigenfaces, é possível citar algumas que foram implementadas neste projeto:

- **Aproximação e Compressão de Imagens**: Projetar imagens sobre o subespaço gerado pelas autofaces nos permite gerar aproximações de imagens que estejam em nossa base de dados. Deependendo da imagem e da base de dados utilizada, as aproximações podem ser satisfatórias com poucas autofaces. Quanto mais autofaces usarmos, mais fiel será a aproximação. Também podemos aproximar imagens que não estão na base de dados, sacrificando um pouco da fidelidade. No código, isso é realizado pela função `approximate_image`.

- **Classificação de Imagens**: Eigenfaces também nos permitem classificar imagens e decidir se dois rostos pertencem ou não à mesma pessoa. Isso pode ser visualizado em um espaço bidimensional projetando as imagens em duas autofaces diferentes. No código, isso é realizado pela função `images_classification`.

- **Reconhecimento Facial**: É possível comparar a imagem do rosto de uma pessoa com as imagens na base de dados utilizada analisando apenas os tamanhos de suas projeções sobre as autofaces e calculando distância entre estes vetores. A imagem cujos tamanhos de projeção apresentar a menor distância em relação à original será a melhor aproximação. Quanto mais autofaces forem utilizadas no cálculo, mais precisa e mais demorada será a aproximação. No código, isso é realizado pela função `person_recognition`.

## Como testar o projeto com suas próprias imagens

- (OPCIONAL) Crie um ambiente virtual *.venv* e o acesse via comandos: **UNIX** ```source .venv/bin/activate``` **Windowns** ```Scripts/activate```
- Instale as bibliotecas por meio do arquivo `requirements.txt` utilizando o comando `pip install -r /..../requirements.txt`
- Crie as pastas para armazenar as imagens que serão usadas no projeto e importe as imagens (provenientes de bases de dados, por exemplo)
- Altere as variáveis que armazenam os caminhos para pastas e arquivos para os caminhos corretos

![image](https://github.com/user-attachments/assets/e696d15e-7eb6-4d9c-8e9e-588de8b571df)

- No final do arquivo, descomente as chamadas de funções que você deseja testar e passe os parâmetros especificados

![image](https://github.com/user-attachments/assets/c0444f82-1e02-43ee-8950-f7f8ee05d124)

- Rode a aplicação com `python main.py` ou `py main.py`

Exemplo de saída:

![image](https://github.com/user-attachments/assets/f79a7e40-f4ea-4dc4-93d9-558be2053aab)
