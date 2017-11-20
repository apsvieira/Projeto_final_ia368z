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

# Referências
[1] https://arxiv.org/pdf/1701.07875.pdf
[2] https://arxiv.org/pdf/1704.00028.pdf
[3] Implementação do WGAN https://github.com/martinarjovsky/WassersteinGAN
[4] Implementação do Improved WGAN https://github.com/caogang/wgan-gp
