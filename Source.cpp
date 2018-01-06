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
#include "Test.h"

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
		
		NN::CMatrix X(2, 400);
		NN::CMatrix Y(1, 400);
		
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 400; j++) {
				X[i][j] = TRAIN_X[i][j];
			}
		}

		for (int i = 0; i < 400; i++) {
			Y[0][i] = TRAIN_Y[i];
		}


		NN::CPerceptron nn(
			400, 2, 
			{ 4, 2, 1 }, 
			{ NN::ActivationFunction::E_TANH, NN::ActivationFunction::E_TANH, NN::ActivationFunction::E_SIGMOID},
			0.5
		);

		nn.fit(X, Y);
		auto& prediction = nn.predict(X);
		auto acc_value = accuracy(prediction, Y);

		char outp[100];
		sprintf_s(outp, "accuracy: %lf", acc_value);
		DBOUT(outp);
	}
	catch (NN::CMatrix::InvalidDimensions e) {
		DBOUT("INVALID DIMENSIONS\n");
	}
}