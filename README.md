# Projeto - Construção, Implementação e Consumo de API de Dados Para Machine Learning
## Instruções Para Execução do Projeto 

### Abra o terminal ou prompt de comando e navegue até a pasta onde você colocou os arquivos

### Execute o comando abaixo para criar a imagem Docker

- docker build -t ml-api-image:prj .

### Execute o comando abaixo para criar o container Docker

- docker run -dit --name ml-api -p 3310:3310 ml-api-image:prj

### Visualize os logs para verificar se a API já foi inicializada

- docker logs ml-api

### Então execute a chamada cliente

- python client.py