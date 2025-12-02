#include <iostream>
#include "neural.hpp"

int main()
{
  MLP mlp(3, 5, 2, 0.01, 1000);
  // Chega o mouse na frente do init_weights pra ver o que aparece.
  mlp.init_weights();
  // Clica do lado do número da linha pra botar uma bolinha vermelha e roda no modo debug pra ver o que acontece.

  std::cout << "Hello, Ebsments!" << std::endl;
  return 0;
}
