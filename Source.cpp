#include <iostream>
#include <memory>
#include <ctime>
#include <sstream>
#include <iomanip>


#include <Windows.h>
#define DBOUT( s )            \
{                             \
   std::ostringstream os_;    \
   os_ << s;                   \
   OutputDebugStringA( os_.str().c_str() );  \
}

#include "Matrix.h"
#include "Perceptron.h"
#include "Activation.h"
#include "Data.h"

double accuracy(const NN::CMatrix& y_pred, const NN::CMatrix& y_real) {
	int correct = 0;
	for (int i = 0; i < y_pred.ncol; ++i) {
		if ((int)y_pred[0][i] == y_real[0][i]) {
			correct++;
		}
	}

	return (double)correct / y_pred.ncol;
}

int main() {
	try {
		auto& X = flowerX_to_matrix();
		auto& Y = flowerY_to_matrix();


		NN::CPerceptron nn(
			X.ncol, // observations 
			X.nrow,  // variables
			2,   // classes
			{ 10, 5, 2 }, // neurons in each layer
			{ NN::ActivationFunction::E_TANH, NN::ActivationFunction::E_RELU, NN::ActivationFunction::E_TANH } // activation functions for each neuron
		);

		nn.set_lr(0.999);
		nn.set_lr_decay_state(true);

		nn.fit(X, Y, 5000);
		auto& prediction = nn.predict(X);
		auto acc_value = accuracy(prediction, Y);

		char outp[100];
		sprintf_s(outp, "accuracy: %lf", acc_value);
		DBOUT(outp);
	}
	catch (NN::CMatrix::InvalidDimensions) {
		DBOUT("INVALID DIMENSIONS\n");
	}
}