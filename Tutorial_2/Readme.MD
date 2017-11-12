Estamos aplicando o tutorial presente no site:

https://github.com/jayleicn/animeGAN

# AMANHÃ TESTAR A FUNÇÃO PARA ARRUMAR OS ERROS!

# To-do list:

* () - 1) Importante é conseguir extrair o paralelismo de GPU's, pois o treinamento total é muito grande.

A questão aqui é que temos multi-GPU's no server Azure, mas no cluster local "brain" apenas uma. Fica esta parte para depois

* () - 2) Melhorar a função de treinamento para ficar mais legível;

Já fiz o escalte do que a função está fazendo, precisa entender alguns métodos que estão se passando ali e quais nomes de variáveis serão utilizados.

* () - 3) Melhorar a função de treinamento para usar a biblioteca do professor;

Não é importante no momento.

* () - 4) Entender o uso de diferentes hiperparametros, para poder ir para outro código.

Isso é muito importante, após arrumar a função e entender o opt.mode == 2, faremos isso.


# Currently doing:


Atualmente mudando a função de treino:

a) Melhorar o print, para ficar apenas em uma linha e não printar a cada batch;

b) Implementar a questão de colocar dois laços dentro da mesma época e arrumar a questão de como calcular a MÉDIA;

c) Trocar as variáveis para nomes que explicam melhor.

d) Implementar para salvar também as figuras reais e não só as fakes, mas não sei se isso é importante, é um teste a se fazer.

e) Entender esta questão de criar apenas uma vez o input e o label, e usar métodos de trocar os seus conteúdos (fill) e qual a vantagem nisso.

f) Descobrir o que é Backward com o Retain Values = True

g) Descobrir pq o detatch dele está em outra parte em questão ao tutorial do toy example (root/Projeto.ipynb)]

h)Testar o opt.model == 2 e ver o q isso significa.

i) 
