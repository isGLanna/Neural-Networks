#pragma once

#include <vector>
#include <random>
using namespace std;

/// @brief A Multi-Layer Perceptron (MLP) neural network class.
class MLP
{
  // Público sempre primeiro, porque é a primeira coisa que o desenvolvedor vai olhar.
public:
  MLP(int entradas, int ocultas, int saidas, double taxa_aprendizado, int epocas);

  /// @brief Applies the ReLU activation function to the input.
  /// @param x
  /// @return
  double relu(double x);

  /// @brief Calculates the derivative of the ReLU activation function.
  /// @param x
  /// @return
  double relu_derivative(double x);

  /// @brief Applies the sigmoid activation function to the input.
  /// @param x
  /// @return
  double sigmoid(double x);

  /// @brief Calculates the derivative of the sigmoid activation function.
  /// @param fx
  /// @return
  double sigmoid_derivative(double fx);

  /// @brief 1 - Atribui pesos para cada conexão entre camadas
  void init_weights();

  /// @brief 2 - Propagação para frente
  vector<double> feedforward(
      const vector<double> &entrada,
      vector<double> &ocultas_ativadas,
      vector<double> &saidas_ativadas);

  /// @brief 6 - função de retorno do erro para ajustar os pesos
  void backpropagation(
      const vector<double> &entrada,
      const vector<double> &real,
      vector<double> &ocultas_ativadas,
      vector<double> &saidas_ativadas);

  /// @brief 3 - função de correção dos pesos
  void update_weights(
      const vector<double> &entradas,
      const vector<double> &ocultas,
      const vector<double> &gradiente_saida,
      const vector<double> &gradiente_oculta);

  /// @brief 5 - função de calculo do erro
  double average_squared_error(const vector<double> &real, const vector<double> &estimated);

private:
  int inputs;
  int hidden;
  int outputs;
  double learning_rate;
  int epocs;

  vector<vector<double>> input_hidden_weights;
  vector<vector<double>> output_hidden_weights;

  vector<double> bias_hidden;
  vector<double> bias_output;

  std::mt19937 rng;
};
