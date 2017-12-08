# Testes mais teóricos sobre Gans


### Tem quatro diferenças principais entre a rede que testamos e a rede que deu certo do Lotufo:

* - 1) Nós vamos até ngf*8 e ele vai de ngf * 4.

* - 2) A rede dele tem um viee (-1,1).squeeze(), tenho de testar o que este reshape está fazendo de verdade

* - 3) A rede D nossa tem: (convt1 a convt 4) -> kernel  4, stride = 1 e padding = 0 e (final conv) -> kernel = 4, stride = 2, padding = 

       A rede D do Lotufo tem: (convt1 e convt 2) -> kernel  4, stride = 1 e padding = 1. (convt3 e convt 4) -> kernel = 4, stride = 1, padding = 0)

* - 4)  A rede G do lotufo tem Relu e a nossa tem Leaky-Relu

## Testes que serão realizados:

* - 1) Um código com os mesmos parâmetros, mas uma cada a menos na nossa rede.

* - 2) Um código com os mesmos parâmetros, mas com activation layers de RELU n nossa rede G

* - 3) Um código com os mesmos parâmetros, mas com os kernels igual ao professor

* - 4) Um código igual ao nosso, mas com o squeeze.  ** Primeiro vou ver se é necessário.

## To-do-list:

- Achei um erro gritante no que eu estava fazendo. A rede para deixar com 64,64 a imagem era meio estranha. Vou ver o jeito canônico de ir dividindo ao meio a figura. Para ir de 28 -> 14 -> 7 -> 4 -> 1!
       