#include "neural.hpp"

#include <iostream>
#include <cmath>
#include <chrono>

using namespace std;

MLP::MLP(int entradas, int ocultas, int saidas, double taxa_aprendizado, int epocas)
    : inputs(entradas), hidden(ocultas), outputs(saidas), learning_rate(taxa_aprendizado)
{
  this->rng.seed((unsigned)chrono::high_resolution_clock::now().time_since_epoch().count());

  this->input_hidden_weights.assign(entradas, vector<double>(ocultas));
  this->output_hidden_weights.assign(ocultas, vector<double>(saidas));
  this->bias_hidden.assign(ocultas, 0.0);
  this->bias_output.assign(saidas, 0.0);
  this->epocs = epocas;

  this->init_weights();
}

double MLP::relu(double x) { return x > 0 ? x : 0; }

double MLP::relu_derivative(double x) { return x > 0 ? 1.0 : 0.0; }

double MLP::sigmoid(double x) { return 1 / (1 + exp(-x)); }

double MLP::sigmoid_derivative(double fx) { return fx * (1 - fx); }

void MLP::init_weights()
{

  uniform_real_distribution<double> dist(-1.0, 1.0);

  for (int i = 0; i < inputs; ++i)
    for (int j = 0; j < hidden; ++j)
      input_hidden_weights[i][j] = dist(rng);

  for (int j = 0; j < hidden; ++j)
    for (int k = 0; k < outputs; ++k)
      output_hidden_weights[j][k] = dist(rng);

  for (int j = 0; j < hidden; ++j)
    bias_hidden[j] = dist(rng);
  for (int k = 0; k < outputs; ++k)
    bias_output[k] = dist(rng);
}

vector<double> MLP::feedforward(
    const vector<double> &entrada,
    vector<double> &ocultas_ativadas,
    vector<double> &saidas_ativadas)
{

  ocultas_ativadas.assign(this->hidden, 0.0);
  saidas_ativadas.assign(this->outputs, 0.0);

  // Calcular ativações da camada oculta (intermediárias)
  for (int j = 0; j < hidden; j++)
  {

    double soma = 0.0;

    for (int i = 0; i < inputs; i++)
    {
      soma += entrada[i] * input_hidden_weights[i][j];
    }

    soma += bias_hidden[j];

    // Aplica ReLU para ativar o neurônio, retornando 1 caso a soma seja positiva diferente de 0
    ocultas_ativadas[j] = this->relu(soma);
  }

  // Calcular ativações da camada de saída (resultado)
  for (int k = 0; k < outputs; k++)
  {

    double soma = 0.0;

    for (int j = 0; j < hidden; j++)
    {
      soma += ocultas_ativadas[j] * output_hidden_weights[j][k];
    }

    soma += bias_output[k];
    saidas_ativadas[k] = this->sigmoid(soma);
  }

  return saidas_ativadas;
}

void MLP::backpropagation(
    const vector<double> &entrada,
    const vector<double> &real,
    vector<double> &ocultas_ativadas,
    vector<double> &saidas_ativadas)
{

  // Executa feedforward para ter os valores atualizados
  feedforward(entrada, ocultas_ativadas, saidas_ativadas);

  vector<double> gradiente_saida(this->outputs);

  // Gradiente da camada de saída
  vector<double> erro_saida(this->outputs);
  for (int k = 0; k < outputs; k++)
  {

    // Calcula o erro nas saídas
    erro_saida[k] = real[k] - saidas_ativadas[k];
    gradiente_saida[k] = erro_saida[k] * sigmoid_derivative(saidas_ativadas[k]);
  }

  vector<double> gradiente_oculta(hidden);

  for (int j = 0; j < hidden; j++)
  {
    double soma = 0.0;

    for (int k = 0; k < this->outputs; k++)
    {
      soma += gradiente_saida[k] * output_hidden_weights[j][k];
    }

    gradiente_oculta[j] = soma * relu_derivative(ocultas_ativadas[j]);
  }

  update_weights(entrada, ocultas_ativadas, gradiente_saida, gradiente_oculta);
}

void MLP::update_weights(
    const vector<double> &entradas,
    const vector<double> &ocultas,
    const vector<double> &gradiente_saida,
    const vector<double> &gradiente_oculta)
{

  // correção de pesos para as camadas ocultas
  for (int j = 0; j < ocultas.size(); j++)
  {
    for (int k = 0; k < outputs; k++)
    {
      double ajuste_peso = learning_rate * gradiente_saida[k];

      output_hidden_weights[j][k] += ajuste_peso;
    }

    // atualiza bias
    bias_output[j] += learning_rate * gradiente_saida[j];
  }

  // correção de pesos para as camadas8 de entrada
  for (int i = 0; i < entradas.size(); i++)
  {
    for (int j = 0; j < ocultas.size(); j++)
    {
      double ajuste_peso = learning_rate * gradiente_oculta[j];

      input_hidden_weights[j][i] += ajuste_peso;
    }
  }

  // Ajusta o bias para cada camada oculta
  for (int j = 0; j < ocultas.size(); j++)
    bias_hidden[j] += learning_rate * gradiente_oculta[j];
}

double MLP::average_squared_error(const vector<double> &real, const vector<double> &estimated)
{
  double erro = 0.0;
  for (size_t i = 0; i < real.size(); i++)
  {
    erro += pow(real[i] - estimated[i], 2) / 2;
  }
  return erro;
}
