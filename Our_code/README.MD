# Diferenças entre o nosso código e do professor Lotufo

* () - Label do prof. lotufo é apenas uma coluna e o nosso são 10 (para o MNIST)

* () - Rede D dá só uma resposta e a nossa dá 11 colunas (10 classes + 1 fake)

* () - Por causa do item acima ele usa a binary cross entropy e nós a categorial cross entropy

* () - Batchsize nosso é de 64 e o dele é de 50

* () - Usamos label_smooth e ele não usa


# To-do list final:


* () - Deixar o codigo classifier_dcgan_review.ipynb funcionando igual ao do Prof. Lotufo,
    mesmo sabendo que ele não vai ser mais um classifier.

* () - Deixar o código do lotufo com a mesma cara do nosso notebook

* () - Salvar as figuras do mnist certo e função plot do Lotufo

* () - Aplicar o mesmo note ao CIFAR-10

* () - Caso não der muito erros ler o codigo do cDCGAN junto com o paper do cDCGAN

* () - Passar a a implementação do cDCGAN para a minha maneira de implementar

## Ver se não dá para fazer figuras com o fixed_noise igual para o dataset do ANIME :)