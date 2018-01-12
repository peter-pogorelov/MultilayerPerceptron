#pragma once

#include "Activation.h"
#include "Random.h"

namespace NN {
	class CAbstractLayer {
	public:
		std::shared_ptr<IInitializer> initializer;
		std::shared_ptr<IActivation> activation_func;

		CMatrix A, Z, W, B;
		CMatrix dA, dZ, dW, dB;

	public:
		CAbstractLayer(int neurons_prev, int neurons, int observations, ActivationFunction activation, int seed = 42) {
			if (activation == ActivationFunction::E_SIGMOID) {
				activation_func = std::make_shared<NN::CSigmoid>();
			}

			else if (activation == ActivationFunction::E_TANH) {
				activation_func = std::make_shared<NN::CTanh>();
			}

			else if (activation == ActivationFunction::E_RELU) {
				activation_func = std::make_shared<NN::CReLu>();
			}

			initializer = std::make_shared<NN::CBengioInitialization>(
				seed, activation_func->get_layer_initialize_constant(neurons_prev, neurons)
			);

			this->W = CMatrix(neurons, neurons_prev, false);
			this->dW = CMatrix(neurons, neurons_prev, initializer);

			this->B = CMatrix(neurons, 1, false);
			this->dB = CMatrix(neurons, 1, nullptr);

			this->Z = CMatrix(neurons, observations, false);
			this->dZ = CMatrix(neurons, observations, nullptr);

			this->A = CMatrix(neurons, observations, false);
			this->dA = CMatrix(neurons, observations, nullptr);
		}

		virtual ~CAbstractLayer() {}

		virtual void forward(CMatrix &input, int iteration) = 0;
		virtual void backward(CAbstractLayer* prev_layer, const CMatrix& x, const CMatrix& y) = 0;
		virtual void update(double lr) = 0;
	};

	class CLayer : public CAbstractLayer {
	public:

		CLayer(int neurons_prev, int neurons, int observations, ActivationFunction activation) :
			CAbstractLayer(neurons_prev, neurons, observations, activation)
		{
		}

		virtual void forward(CMatrix &input, int iteration);
		virtual void backward(CAbstractLayer* prev_layer, const CMatrix& x, const CMatrix& y);
		virtual void update(double lr);
	};

	class CRidgeLayer : public CLayer {
	private:
		double penalty;
	public:
		CRidgeLayer(int neurons_prev, int neurons, int observations, ActivationFunction activation, double penalty) :
			CLayer(neurons_prev, neurons, observations, activation)
		{
			this->penalty = penalty;
		}

		virtual void backward(CAbstractLayer* prev_layer, const CMatrix& x, const CMatrix& y);
	};

	class CDropoutLayer : public CLayer {
	private:
		double do_proba;
	public:
		CDropoutLayer(int neurons_prev, int neurons, int observations, ActivationFunction activation, double do_proba) :
			CLayer(neurons_prev, neurons, observations, activation)
		{
			this->do_proba = do_proba;
		}

		virtual void forward(CMatrix &input, int iteration);
	
	private:
		virtual CMatrix generate_dropout(int seed, int neurons);
	};

	class CCrossEntropyLayer : public CLayer
	{
	public:
		CCrossEntropyLayer(int neurons_prev, int neurons, int observations, ActivationFunction activation) :
			CLayer(neurons_prev, neurons, observations, activation)
		{
		}

		virtual void backward(CAbstractLayer* prev_layer, const CMatrix& x, const CMatrix& y);
	};
}