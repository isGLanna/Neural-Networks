#include <iostream>
#include "neural.hpp"

#include <lib_aleatoria/lib.hpp>

int main()
{
  MLP mlp(3, 5, 2, 0.01, 1000);
  // Chega o mouse na frente do init_weights pra ver o que aparece.
  mlp.init_weights();
  // Clica do lado do número da linha pra botar uma bolinha vermelha e roda no modo debug pra ver o que acontece.

  // Isso foi chamado de uma biblioteca de dentro da pasta include
  lib_aleatoria();

  std::cout << "Hello, Ebsments!" << std::endl;
  return 0;
}
