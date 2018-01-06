#pragma once
#include <ctime>
#include <cmath>

namespace NN {
	class IInitializer {
	protected:
		int param = 0;
	public:
		IInitializer(int param) : param(param) {}
		virtual ~IInitializer() {}

		virtual void initialize(double* arr, int size) = 0;
	};

	class CRandomInitializer : public IInitializer {
	public:
		CRandomInitializer(int seed) : IInitializer(seed) {
			std::srand(seed);
		}

		void initialize(double* arr, int size) {
			for (int i = 0; i < size; ++i)
				arr[i] = (float)std::rand() / RAND_MAX;
		}
	};

	class CConstInitializer : public IInitializer {
	public:
		CConstInitializer(int value) : IInitializer(value) {}

		void initialize(double* arr, int size) {
			for (int i = 0; i < size; ++i)
				arr[i] = param;
		}
	};
}