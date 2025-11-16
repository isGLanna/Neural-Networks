// importar bibliotacas básicas do c++
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
using namespace std;

class MLP {
  private:
    int entradas;
    int ocultas;
    int saidas;
    double taxa_aprendizado;
    int epocas;

    vector<vector<double>> pesos_entradas_para_ocultas;
    vector<vector<double>> pesos_ocultas_para_saidas;

    vector<double> bias_ocultas;
    vector<double> bias_saidas;

    std::mt19937 rng;

  public:
    MLP(int entradas, int ocultas, int saidas, double taxa_aprendizado, int epocas)
        : entradas(entradas), ocultas(ocultas), saidas(saidas), taxa_aprendizado(taxa_aprendizado)
    {
        this->rng.seed((unsigned) chrono::high_resolution_clock::now().time_since_epoch().count());

        this->pesos_entradas_para_ocultas.assign(entradas, vector<double>(ocultas));
        this->pesos_ocultas_para_saidas.assign(ocultas, vector<double>(saidas));
        this->bias_ocultas.assign(ocultas, 0.0);
        this->bias_saidas.assign(saidas, 0.0);
        this->epocas = epocas;

        this->inicializar_pesos();
    }

    // função de ativação ReLU
    double relu(double x) {return x > 0 ? x : 0;}

    double relu_derivada(double x) { return x > 0 ? 1.0 : 0.0; }

    // Função de ativação sigmóide
    double sigmoid (double x) {return 1 / (1 + exp(-x));}

    double sigmoid_derivada(double fx) {return fx * (1 - fx);}

    // 1 - Atribui pesos para cada conexão entre camadas
    void inicializar_pesos() {

      uniform_real_distribution<double> dist(-1.0, 1.0);
      
      for (int i = 0; i < entradas; ++i)
        for (int j = 0; j < ocultas; ++j)
          pesos_entradas_para_ocultas[i][j] = dist(rng);

      for (int j = 0; j < ocultas; ++j)
        for (int k = 0; k < saidas; ++k)
          pesos_ocultas_para_saidas[j][k] = dist(rng);

      for (int j = 0; j < ocultas; ++j) bias_ocultas[j] = dist(rng);
      for (int k = 0; k < saidas; ++k)  bias_saidas[k]  = dist(rng);
    }

    // 2 - Propagação para frente
    vector<double> feedforward(
      const vector<double>& entrada,
      vector<double>& ocultas_ativadas,
      vector<double>& saidas_ativadas) {

      ocultas_ativadas.assign(this->ocultas, 0.0);
      saidas_ativadas.assign(this->saidas, 0.0);

      // Calcular ativações da camada oculta (intermediárias)
      for (int j = 0; j < ocultas; j++) {

        double soma = 0.0;

        for (int i = 0; i < entradas; i++){
          soma += entrada[i] * pesos_entradas_para_ocultas[i][j];
        }

        soma += bias_ocultas[j];

        // Aplica ReLU para ativar o neurônio, retornando 1 caso a soma seja positiva diferente de 0
        ocultas_ativadas[j] = this->relu(soma);
      }

      // Calcular ativações da camada de saída (resultado)
      for (int k = 0; k < saidas; k++) {
        
        double soma = 0.0;

        for (int j = 0; j < ocultas; j++) {
          soma += ocultas_ativadas[j] * pesos_ocultas_para_saidas[j][k];
        }

        soma += bias_saidas[k];
        saidas_ativadas[k] = this->sigmoid(soma);
      }

      return saidas_ativadas;
    }


    // 6 - função de retorno do erro para ajustar os pesos
    void backpropagation(
      const vector<double> &entrada, 
      const vector<double> &real,
      vector<double>& ocultas_ativadas,
      vector<double>& saidas_ativadas){

      // Executa feedforward para ter os valores atualizados
      feedforward(entrada, ocultas_ativadas, saidas_ativadas);

      vector<double> gradiente_saida(this->saidas);

      // Gradiente da camada de saída
      vector<double> erro_saida(this->saidas);
      for (int k = 0; k < saidas; k++) {

        // Calcula o erro nas saídas
        erro_saida[k] = real[k] - saidas_ativadas[k];
        gradiente_saida[k] = erro_saida[k] * sigmoid_derivada(saidas_ativadas[k]);
      }

      vector<double> gradiente_oculta(ocultas);

      for (int j = 0; j < ocultas; j++) {
        double soma = 0.0;

        for (int k = 0; k < this->saidas; k++) {
          soma += gradiente_saida[k] * pesos_ocultas_para_saidas[j][k];
        }

        gradiente_oculta[j] = soma * relu_derivada(ocultas_ativadas[j]);
      }

      update_weights(entrada, ocultas_ativadas, gradiente_saida, gradiente_oculta);
    }


    // 3 - função de correção dos pesos
    void update_weights(
      const vector<double>& entradas, 
      const vector<double>& ocultas, 
      const vector<double>& gradiente_saida ,
      const vector<double>& gradiente_oculta) {
      
      // correção de pesos para as camadas ocultas
      for (int j = 0; j < ocultas.size(); j++) {
        for (int k = 0; k < saidas; k++) {
          double ajuste_peso = taxa_aprendizado * gradiente_saida[k];
          
          pesos_ocultas_para_saidas[j][k] += ajuste_peso;
        }
        
        // atualiza bias
        bias_saidas[j] += taxa_aprendizado * gradiente_saida[j];
      }

      // correção de pesos para as camadas8 de entrada
      for (int i = 0; i < entradas.size(); i++) {
        for (int j = 0; j < ocultas.size(); j++) {
          double ajuste_peso = taxa_aprendizado * gradiente_oculta[j];
          
          pesos_entradas_para_ocultas[j][i] += ajuste_peso;
        }
      }

      // Ajusta o bias para cada camada oculta
      for (int j = 0; j < ocultas.size(); j++) 
        bias_ocultas[j] += taxa_aprendizado * gradiente_oculta[j];
    }

    // 5 - função de calculo do erro
    double erro_quadratico_medio(const vector<double>& real, const vector<double>& estimado) {
      double erro = 0.0;
      for (size_t i = 0; i < real.size(); i++) {
        erro += pow(real[i] - estimado[i], 2) / 2;
      }
      return erro;
    }
};