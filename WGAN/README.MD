Comentários sobre o paper "Wasserstein GAN" [1, 3] e sobre as modificações
propostas em "Improved Training of Wasserstein GANs" [2, 4].



# Ideia:
  A intenção do trabalho é encontrar uma função de perda que evite problemas de
  instabilidade no treinamento de GANs e que forneça mais informação sobre a
  convergência dos modelos. Não é um modelo em si, é uma forma de treinar.

  Proposta é usar a distância "Earth Mover" (Wasserstein-1, daí o nome).

# Dúvidas:
  1. Deveria ser aplicado o weights_init para qlqr configuração do D? (line 124)

  Atualmente só é aplicado caso não seja a MLP, mas para o G todas as redes são
  inicializadas com essa função.

  2. No treinamento, é feito um "requires_grad=False/True". Isso não poderia ser
  substituído por train/eval?

  3. Qual a diferença entre fazer .backward() e fazer .backward(one)?

  É usado .backward(one) para os dados reais e .backward(mone) para fake no D.
  No G, é usado .backward(one) para os dados falsos.

# Algoritmo (WGAN):
  1. Durante D_iters iterações:
      1) Pega 'batch_size' samples reais
      2) Calcula errD_real =  netD(real)
      3) Faz backpropagation (ver duvida 3)
      4) Gera 'bacth_size' samples falsos
      5) Calcula errD_fake = netD(fake)
      6) Faz backpropagation (ver duvida 3)
      7) Calcula a perda total errD = errD_real - errD_fake
      8) Passo do otimizador de D é dado
  2. Gera 'batch_size' samples falsos
  3. Calcula errG = netD(fake)
  4. Faz backpropagation (ver duvida 3)
  5. Passo do otimizador de G é dado

# Sobre o código:
- Otimizador usado por padrão é o RMSprop
- Número de filtros das convoluções e tamanho da rede são calculados a partir
do tamanho de imagem utilizado
- O número de iterações do D nas primeiras épocas é muito maior (100 vs 5)
para tentar garantir que o discriminador sempre esteja em um ponto ótimo
- Esse processo também é repetido a cada 500 épocas para melhorar resultados
- As losses são impressas e são relevantes para analisar a convergência

# Diferenças do Improved WGAN
  Ideia do trabalho é que o clipping dos pesos afeta negativamente a convergência
  e seria uma má forma de garantir a Lipschitz-continuidade. Proposta é usar
  outro método para garantir que os pesos não explodam ou zerem, sem limitar
  expressamente os valores.
  
  1. Ao invés de usar clipping para os pesos, introduz penalidade no gradiente
  2. Substitui BatchNorm por LayerNorm
  3. Usa ADAM (lr=1e-4,beta1=0, beta2=0.9) ao invés de RMSprop(lr=5e-5)

# Diferenças na Implementação:
  1. Redes são definidas estaticamente, uma para cada Dataset
  2. Encapsula várias funções para simplificar o código
  3. A função de calcular penalidade realiza interpolação dos dados reais com os
  imaginários e os passa de novo pela rede.

# Dúvidas sobre essa implementação:
  1. O coeficiente de interpolação precisa ser aleatório? Paper usa 0.5 fixo.
  2. Esse é o jeito mais fácil de implementar a penalidade? Não é possível manter
  os gradientes e fazer a interpolação com os que já foram calculados? (rapidez)

# Comentários aleatórios e coisas pras quais provavelmente não temos tempo:
  1. Há outra proposta de penalidade para os gradientes, em [5]
  2. Em [6] é proposto usar uma distância diferente. Resultados similares a WGAN

# Referências
[1] Wasserstein GAN https://arxiv.org/pdf/1701.07875.pdf

[2] Improved Training of Wasserstein GANs https://arxiv.org/pdf/1704.00028.pdf

[3] Implementação do WGAN https://github.com/martinarjovsky/WassersteinGAN

[4] Implementação do Improved WGAN https://github.com/caogang/wgan-gp

[5] On the regularization of WGANs https://arxiv.org/pdf/1709.08894.pdf

[6] The Cramer Distance as a Solution to Biased Wasserstein Gradients https://arxiv.org/pdf/1705.10743.pdf
