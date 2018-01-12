#include "Random.h"

int bernoulli(double proba)
{
	static const int high = 10000;
	register int rand_value = std::rand() % high;
	return rand_value > proba * high;
}