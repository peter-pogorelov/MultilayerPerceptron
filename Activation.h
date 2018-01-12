#pragma once
#include <functional>

#include "Matrix.h"

namespace NN {
	enum class ActivationFunction {
		E_SIGMOID,
		E_TANH,
		E_RELU
	};


	class IActivation {
	protected:
		std::function<double(double)> func;
		std::function<double(double)> derivative_func;
	public:
		IActivation() {}
		virtual ~IActivation() {}

		CMatrix operator()(const CMatrix& a) {
			CMatrix result(a.nrow, a.ncol);

			for (int i = 0; i < a.nrow; ++i){
				for (int j = 0; j < a.ncol; ++j) {
					result[i][j] = this->func(a[i][j]);
				}
			}

			return result;
		}

		CMatrix derivative(const CMatrix& a) {
			CMatrix result(a.nrow, a.ncol);

			for (int i = 0; i < a.nrow; ++i) {
				for (int j = 0; j < a.ncol; ++j) {
					result[i][j] = this->derivative_func(a[i][j]);
				}
			}

			return result;
		}

		virtual double get_layer_initialize_constant(int input, int output) = 0;
	};

	class CSigmoid : public IActivation {
	public:
		CSigmoid() {
			this->func = std::function<double(double)>([=](double a)-> double {
				return 1. / (1. + exp(-a));
			});

			this->derivative_func = std::function<double(double)>([=](double a) -> double {
				double sigmoid = this->func(a);
				return sigmoid * (1 - sigmoid);
			});
		}

		virtual double get_layer_initialize_constant(int input, int output) {
			double benigo = sqrt(6. / (input + output));

			return 4 * benigo;
		}
	};

	class CTanh : public IActivation {
	public:
		CTanh() {
			this->func = std::function<double(double)>([=](double a)-> double {
				return std::tanh(a);
			});

			this->derivative_func = std::function<double(double)>([=](double a) -> double {
				double th = std::tanh(a);
				return 1 - th * th;
			});
		}

		virtual double get_layer_initialize_constant(int input, int output) {
			double benigo = sqrt(6. / (input + output));

			return benigo;
		}
	};

	class CReLu : public IActivation {
	public:
		CReLu() {
			this->func = std::function<double(double)>([=](double a) -> double {
				return a ? a > 0 : 0;
			});

			this->derivative_func = std::function<double(double)>([=](double a) -> double {
				return 1 ? a > 0 : 0;
			});
		}

		virtual double get_layer_initialize_constant(int input, int output) {
			double benigo = sqrt(6. / (input + output));

			return benigo;
		}
	};
}