#pragma once
#include <ctime>
#include <cmath>
#include <random>

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

	class CBengioInitialization : public IInitializer {
	private:
		std::default_random_engine generator;
		std::uniform_real_distribution<double> distribution;
		double current_benigo;
	public:
		CBengioInitialization(int seed, double initialize_constant) : IInitializer(seed) {
			generator.seed(seed);
			current_benigo = initialize_constant;
			distribution = std::uniform_real_distribution<double>(-current_benigo, current_benigo);
		}

		inline double random() {
			return distribution(generator);
		}

		void initialize(double* arr, int size) {

			for (int i = 0; i < size; ++i)
				arr[i] = this->random();
		}
	};
}