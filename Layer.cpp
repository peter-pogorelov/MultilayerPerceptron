#include <cmath>
#include "Layer.h"

#include <Windows.h>
#define DBOUT( s )            \
{                             \
   std::ostringstream os_;    \
   os_ << s;                   \
   OutputDebugStringA( os_.str().c_str() );  \
}

namespace NN {
	void CLayer::forward(CMatrix &input, int iteration) {
		this->Z = this->W.dot(input);
		this->Z = this->Z.add_vector(this->B);
		this->A = (*this->activation_func)(this->Z);
	}

	void CLayer::backward(CAbstractLayer* prev_layer, const CMatrix& x, const CMatrix& y) {
		double m = (double)x.ncol;

		this->dZ = this->dA * this->activation_func->derivative(this->Z);
		this->dB = this->dZ.sum(1) * (1 / m);

		if (prev_layer) {
			this->dW = this->dZ.dot(prev_layer->A.transpose()) * (1 / m);
			prev_layer->dA = this->W.transpose().dot(this->dZ);
		}
		else {
			this->dW = this->dZ.dot(x.transpose()) * (1 / m);
		}
	}

	void CLayer::update(double lr) {
		this->W = this->W - this->dW * lr;
		this->B = this->B.add_vector(-this->dB * lr);
	}

	void CRidgeLayer::backward(CAbstractLayer* prev_layer, const CMatrix& x, const CMatrix& y) {
		double m = (double)x.ncol;

		this->dZ = this->dA * this->activation_func->derivative(this->Z);
		this->dB = this->dZ.sum(1) * (1 / m);

		if (prev_layer) {
			this->dW = this->dZ.dot(prev_layer->A.transpose()) * (1 / m) - this->W * (this->penalty / m);
			prev_layer->dA = this->W.transpose().dot(this->dZ);
		}
		else {
			this->dW = this->dZ.dot(x.transpose()) * (1 / m) - this->W * (this->penalty / m);
		}
	}

	CMatrix CDropoutLayer::generate_dropout(int seed, int neurons) {
		CMatrix dropout_neurons(neurons, 1);
		std::srand(seed); // make sure that each iteration would be with it's own unique seed
		for (int i = 0; i < neurons; ++i) {
			dropout_neurons[i][0] = bernoulli(this->do_proba);
		}

		return dropout_neurons;
	}

	void CDropoutLayer::forward(CMatrix &input, int iteration) {
		if (iteration == -1) {
			this->CLayer::forward(input, iteration); // do not drop units if iteration = -1
			return;
		}
		auto& dropout_neurons = this->generate_dropout(iteration, this->A.nrow);

		this->Z = this->W.dot(input);
		this->Z = this->Z.add_vector(this->B);
		this->A = (*this->activation_func)(this->Z).multiply_vector(dropout_neurons);
	}

	void CCrossEntropyLayer::backward(CAbstractLayer* prev_layer, const CMatrix& x, const CMatrix& y) {
		double m = (double)y.ncol;
		
		this->dZ = this->A - y;
		this->dW = this->dZ.dot(prev_layer->A.transpose()) * (1 / m);
		this->dB = this->dZ.sum(1) * (1 / m);
		prev_layer->dA = this->W.transpose().dot(this->dZ);
	}
}