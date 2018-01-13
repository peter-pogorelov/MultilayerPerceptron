#include <iostream>
#include <vector>
#include <tuple>

/*
#include <Windows.h>
#define DBOUT( s )            \
{                             \
   std::ostringstream os_;    \
   os_ << s;                   \
   OutputDebugStringA( os_.str().c_str() );  \
}
*/

#include "Matrix.h"
#include "Perceptron.h"
#include "Activation.h"
#include "Data.h"
#include "MiniBatch.h"

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
			// omg it appears to be really ugly, but easy to copy / paste to make additional layers
			std::vector<std::tuple<NN::layer_size, NN::ActivationFunction, NN::LayerTypes>> {

				std::tuple<NN::layer_size, NN::ActivationFunction, NN::LayerTypes>\
				(4, NN::ActivationFunction::E_TANH, NN::LayerTypes::E_PLAIN),

			}
		);
		
		nn.set_lr(0.999);
		nn.set_lr_decay_state(true);
		nn.set_train_method(std::make_shared<NN::CMiniBatch>(128, 42));

		nn.fit(X, Y, 1000);

		auto& prediction = nn.predict(X);
		auto acc_value = accuracy(prediction, Y);

		std::cout << "accuracy:" << acc_value << std::endl;
		std::cout << "please press any key." << std::endl;
		std::getchar();
		std::getchar();
	}
	catch (NN::CMatrix::InvalidDimensions) {
		std::cout << "invalid dimensions" << std::endl;
	}
	catch (...) {
		std::cout << "something went wrong." << std::endl;
	}
}