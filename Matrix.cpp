#include <memory>

#include "Matrix.h"
#include "MatrixInitializer.h"

#include <Windows.h>
#define DBOUT( s )            \
{                             \
   std::ostringstream os_;    \
   os_ << s;                   \
   OutputDebugStringA( os_.str().c_str() );  \
}

NN::CMatrix::CMatrix() {
	this->matrix = nullptr;
}

NN::CMatrix::CMatrix(int nrow, int ncol, std::shared_ptr<IInitializer> initializer) {
	this->nrow = nrow;
	this->ncol = ncol;

	this->matrix = new double*[this->nrow];

	for (int i = 0; i < this->nrow; ++i) {
		this->matrix[i] = new double[this->ncol];
	}

	if (initializer) {
		this->initializer = initializer;
		this->initialize();
	}
	else {
		this->initializer = std::make_shared<CConstInitializer>(0);
		this->initialize();
	}
}

NN::CMatrix::CMatrix(const NN::CMatrix& me) {
	this->deep_copy_object(me);
}

NN::CMatrix& NN::CMatrix::operator=(const CMatrix & m) {
	for (int i = 0; i < this->nrow; ++i)
		delete[] this->matrix[i];

	delete[] this->matrix;

	this->deep_copy_object(m);
	return *this;
}

NN::CMatrix::~CMatrix() {
	if (this->matrix) {
		for (int i = 0; i < this->nrow; ++i)
			delete[] this->matrix[i];

		delete[] this->matrix;
	}
}
void NN::CMatrix::initialize() {
	if (this->initializer == nullptr) {
		throw InvalidInitializer();
	}

	for (int i = 0; i < this->nrow; ++i) {
		this->initializer->initialize(this->matrix[i], this->ncol);
	}
}

void NN::CMatrix::set_initializer(IInitializer* init) {
	this->initializer = std::shared_ptr<NN::IInitializer>(init);
}

double* NN::CMatrix::operator[](int i) const { 
	return this->matrix[i]; 
}

NN::CMatrix NN::CMatrix::dot(const NN::CMatrix & m) {
	if (this->ncol != m.nrow)
		throw InvalidDimensions();

	NN::CMatrix result(this->nrow, m.ncol, false);

	for (int i = 0; i < this->nrow; ++i) {
		for (int j = 0; j < m.ncol; ++j) {
			double value = 0;
			for (int k = 0; k < this->ncol; ++k) {
				value += this->matrix[i][k] * m.matrix[k][j];
			}

			result.matrix[i][j] = value;
		}
	}

	return result;
}

NN::CMatrix NN::CMatrix::sum(int axis) {
	if (axis == 1) {
		NN::CMatrix result(this->nrow, 1, false);

		for (int i = 0; i < this->nrow; ++i) {
			result.matrix[i][0] = 0;
			for (int j = 0; j < this->ncol; ++j)
				result.matrix[i][0] += this->matrix[i][j];
		}

		return result;
	}
	else if (axis == 0) {
		NN::CMatrix result(1, this->ncol, false);

		for (int i = 0; i < this->ncol; ++i) {
			result.matrix[0][i] = 0;
			for (int j = 0; j < this->nrow; ++j)
				result.matrix[0][i] += this->matrix[j][i];
		}

		return result;
	}

	throw;
}

NN::CMatrix NN::CMatrix::add_vector(const NN::CMatrix& m) {
	if (m.ncol == 1 && m.nrow == this->nrow) {
		NN::CMatrix result = *this;

		// summing over rows
		for (int i = 0; i < this->ncol; ++i) {
			for (int j = 0; j < this->nrow; ++j) {
				result[j][i] += m[j][0];
			}
		}

		return result;
	}
	else if (m.ncol == this->ncol && m.nrow == 1) {
		NN::CMatrix result = *this;

		// summing over rows
		for (int i = 0; i < this->nrow; ++i) {
			for (int j = 0; j < this->ncol; ++j) {
				result[i][j] += m[0][j];
			}
		}

		return result;
	}

	throw InvalidDimensions();
}

NN::CMatrix NN::CMatrix::multiply_vector(const  NN::CMatrix& m) {
	if (m.ncol == 1 && m.nrow == this->nrow) {
		NN::CMatrix result = *this;

		// summing over rows
		for (int i = 0; i < this->ncol; ++i) {
			for (int j = 0; j < this->nrow; ++j) {
				result[j][i] *= m[j][0];
			}
		}

		return result;
	}
	else if (m.ncol == this->ncol && m.nrow == 1) {
		NN::CMatrix result = *this;

		// summing over rows
		for (int i = 0; i < this->nrow; ++i) {
			for (int j = 0; j < this->ncol; ++j) {
				result[i][j] *= m[0][j];
			}
		}

		return result;
	}

	throw InvalidDimensions();
}

NN::CMatrix NN::CMatrix::transpose() const{
	NN::CMatrix result(this->ncol, this->nrow, false);

	for (int i = 0; i < this->nrow; ++i) {
		for (int j = 0; j < this->ncol; ++j) {
			result.matrix[j][i] = this->matrix[i][j];
		}
	}

	return result;
}

NN::CMatrix NN::CMatrix::shuffle(int seed) const {
	CMatrix shuffled = this->transpose();

	std::default_random_engine generator;
	// make sure that batches are the same within the each epoch
	generator.seed(seed);

	std::shuffle(
		shuffled.get(),
		shuffled.get() + shuffled.nrow,
		generator
	);

	return shuffled.transpose();
}

NN::CMatrix NN::CMatrix::vertical_slice(int from, int to) const
{
	CMatrix matr(this->nrow, to-from+1);

	for (int i = 0; i < this->nrow; ++i) {
		for (int j_1 = from, j_2 = 0; j_1 <= to; ++j_1, ++j_2) {
			matr[i][j_2] = this->matrix[i][j_1];
		}
	}

	return matr;
}

NN::CMatrix NN::CMatrix::horizontal_slize(int from, int to) const
{
	CMatrix matr(to - from + 1, this->ncol);

	for (int i_1 = from, i_2 = 0; i_1 < to; ++i_1, ++i_2) {
		for (int j = 0; j < this->ncol; ++j) {
			matr[i_2][j] = this->matrix[i_1][j];
		}
	}

	return matr;
}

NN::CMatrix NN::CMatrix::operator*(double scalar) {
	NN::CMatrix result(this->nrow, this->ncol, false);

	for (int i = 0; i < this->nrow; ++i) {
		for (int j = 0; j < this->ncol; ++j) {
			result.matrix[i][j] = this->matrix[i][j] * scalar;
		}
	}

	return result;
}

NN::CMatrix NN::CMatrix::operator*(const CMatrix & m) {
	if (this->ncol != m.ncol || this->nrow != m.nrow)
		throw InvalidDimensions();

	NN::CMatrix result(this->nrow, this->ncol, false);

	for (int i = 0; i < this->nrow; ++i) {
		for (int j = 0; j < this->ncol; ++j) {
			result.matrix[i][j] = this->matrix[i][j] * m.matrix[i][j];
		}
	}

	return result;
}

NN::CMatrix NN::CMatrix::operator+(const CMatrix & m) {
	if (this->ncol != m.ncol || this->nrow != m.nrow)
		throw InvalidDimensions();

	NN::CMatrix result(this->nrow, this->ncol, false);

	for (int i = 0; i < this->nrow; ++i) {
		for (int j = 0; j < this->ncol; ++j) {
			result.matrix[i][j] = this->matrix[i][j] + m.matrix[i][j];
		}
	}

	return result;
}

NN::CMatrix NN::CMatrix::operator-(const CMatrix & m) {
	if (this->ncol != m.ncol || this->nrow != m.nrow)
		throw InvalidDimensions();

	NN::CMatrix result(this->nrow, this->ncol, false);

	for (int i = 0; i < this->nrow; ++i) {
		for (int j = 0; j < this->ncol; ++j) {
			result.matrix[i][j] = this->matrix[i][j] - m.matrix[i][j];
		}
	}

	return result;
}

NN::CMatrix NN::CMatrix::operator-() {
	NN::CMatrix result(this->nrow, this->ncol, false);

	for (int i = 0; i < result.nrow; ++i) {
		for (int j = 0; j < result.ncol; ++j) {
			result[i][j] = -1 * this->matrix[i][j];
		}
	}

	return result;
}

std::string NN::CMatrix::to_string() {
	std::string sresult = "[\n";
	for (int i = 0; i < this->nrow; i++) {
		std::stringstream srow;
		srow << "[";
		for (int j = 0; j < this->ncol - 1; j++) {
			srow << std::fixed << std::setprecision(5) << this->matrix[i][j];
			srow << "\t,";
		}
		srow << std::fixed << std::setprecision(5) << this->matrix[i][this->ncol - 1];

		sresult += srow.str();
		sresult += "],\n";
	}

	return sresult + "]\n";
}

void NN::CMatrix::deep_copy_object(const NN::CMatrix &m) {
	this->nrow = m.nrow;
	this->ncol = m.ncol;

	this->matrix = new double*[this->nrow];
	this->initializer = m.initializer;

	for (int i = 0; i < this->nrow; ++i) {
		this->matrix[i] = new double[this->ncol];
	}

	for (int i = 0; i < this->nrow; ++i) {
		for (int j = 0; j < this->ncol; ++j) {
			this->matrix[i][j] = m.matrix[i][j];
		}
	}
}

double** NN::CMatrix::get() {
	return this->matrix;
}

void NN::CMatrix::copy_object(const NN::CMatrix &m) {
	this->nrow = m.nrow;
	this->ncol = m.ncol;

	this->initializer = m.initializer;

	for (int i = 0; i < this->nrow; ++i) {
		for (int j = 0; j < this->ncol; ++j) {
			this->matrix[i][j] = m.matrix[i][j];
		}
	}
}