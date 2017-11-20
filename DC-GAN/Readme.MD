Comentários sobre o paper:

https://arxiv.org/pdf/1703.05192.pdf

https://github.com/carpedm20/DiscoGAN-pytorch

# Idéia geral:

    Dá para pensar que o paper funciona como encoderers e decoders, mas ao invés de procurar uma 
    representação de baixa dimensionalidade dos dados de entrada, ele procura uma representação de
    X_A, dado no domínio A, para X_AB, representação de X_A no domínio B.


# Dúvidas:

(X) - Na função de treinamento, aparece isto:
            
            l_d_A = l_d_A_real + l_d_A_fake
            
            l_d_B = l_d_B_real + l_d_B_fake

            l_d = l_d_A + l_d_B

            l_d.backward()


    A dúvida é: "Por que ele não fez l_d_a.backward() e l_d_b.backward()? Talvez essa seja a questão da tal
                aprendida as relações de cross-domain, que o paper fala. Questão conceitual importante que
                precisa ficar esclarecida."
    
    Possível resposta abaixo.

(X) - O mesmo acontece com a rede G:

     l_g = l_gan_A + l_gan_B + l_const_A + l_const_B

     l_g.backward()

     A pergunta é que se só o scalar, "valor da loss" é compartilhado, ou alguma coisa tipo pesos, algo assim.

    <b>Resposta:</b>
    
    Aqui tem a possível resposta. Parece que ele só soma os escalares e propaga naturalmente.
    https://discuss.pytorch.org/t/how-to-use-the-backward-functions-for-multiple-losses/1826/8

# Algoritmo:

    1) Pega um batch X_a e X_b
    2) Zera os gradientes de D
    3) Encoda uma imagem de um domínio para o outro
       X_ab = (GAB(X_a))
       X_ba = (GBA(X_b))
    4) Decoda a imagem, ou encoda a volta da imagem que foi encodada na ida.
       X_aba = GBA(X_ab)
       X_bab = GAB(X_ba) 
    5)  Calcula a perda de reconstrução, o quão bom a encodificação->decodificação foi
        l_const_A = d(x_ABA, x_A)
        l_const_B = d(x_BAB, x_B)
    6) Calcula a perda de Gans, ou seja, o quão bom a primeira encodificação foi
        l_d_A_real, l_d_A_fake = criterion(D_A(x_A), real_tensor), criterion(D_A(x_BA), fake_tensor)
        l_d_B_real, l_d_B_fake = criterion(D_B(x_B), real_tensor), criterion(D_B(x_AB), fake_tensor)

        #loss_ real é o para o discriminator saber uma imagem real (X_a pertence a A) e uma fake (X_ba pertence a A).
        #real_tensor = uma matriz de 1, #fake_tensor = matriz de 0
    
    7) Faz o back-propagation na rede D, com as duas perdas:
        l_d_A = l_d_A_real + l_d_A_fake
        l_d_B = l_d_B_real + l_d_B_fake
        l_d = l_d_A + l_d_B

        l_d.backward()
    8) Zera os gradientes de G
    9) Computa de novo X_ab, X_ba, X_aba, X_bab
    10 ) Calcula a perda de reconstrução, ou seja o quão boa a decoficação está sendo:
        l_const_A = d(x_ABA, x_A)
        l_const_B = d(x_BAB, x_B)
    11) Soma a perda de reconstrução e a perda GAN's, soma e propaga esta perda para as redes G_a e G_b 
        l_g = l_gan_A + l_gan_B + l_const_A + l_const_B
        l_g.backward()

# Remarks sobre o código: 

### Métricas para a função de treinamento

    Eles printam as loss apenas
 

Transforms que ele usa: 
        self.transform = transforms.Compose([
            transforms.Scale(scale_size), 
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

+ um config.skip_pix2pix_processing que são transformações realizadas com uma biblioteca externa
por meio de flags que ele passa.


Inicialização dos pesos!

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

Criação dos modelos:

A jogada é que um encoder dele é o decoder em outra parte (com o outro discriminator) e vice-versa!

    self.G_AB = GeneratorFC(2, 2, [config.fc_hidden_dim] * config.g_num_layer)
    self.G_BA = GeneratorFC(2, 2, [config.fc_hidden_dim] * config.g_num_layer)

    self.D_A = DiscriminatorFC(2, 1, [config.fc_hidden_dim] * config.d_num_layer)
    self.D_B = DiscriminatorFC(2, 1, [config.fc_hidden_dim] * config.d_num_layer)

Tamanho das convolucionais:

  if self.cnn_type == 0:
        #conv_dims, deconv_dims = [64, 128, 256, 512], [512, 256, 128, 64]
        conv_dims, deconv_dims = [64, 128, 256, 512], [256, 128, 64]
  elif self.cnn_type == 1:
        #conv_dims, deconv_dims = [32, 64, 128, 256], [256, 128, 64, 32]
        conv_dims, deconv_dims = [32, 64, 128, 256], [128, 64, 32]

## Função de Treinamento!

    LOSS!
    d = Mean Square Error

    bce = Binary Cross Entropy  

    Optimizer!

    Optimizer do D = Adam Optimizer com weight decay

    Optimzer do G = Adam Optimizer

    Não achei label_smoother no código !


