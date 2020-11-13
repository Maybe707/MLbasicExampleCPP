#include <initializer_list>
#include <iostream>
#include <iterator>
#include <pthread.h>
#include <cassert>
#include <vector>

template<class T, int row, int col>
class Matrix
{
public:
    int matrixRow;
    int matrixCol;
    T matrix[row][col];
    Matrix() : matrixRow(row), matrixCol(col)
    {
    }

    Matrix(std::initializer_list<std::initializer_list<T>> list)
    {
        *this = list;
    }

    Matrix<T, col, row> matrixTranspose()
        {
            Matrix<T, col, row> tempTranceponentMatrix;
            for(int i = 0; i < row; ++i)
                for(int j = 0; j < col; ++j)
                {
                    tempTranceponentMatrix[j][i] = matrix[i][j];
                }
            return tempTranceponentMatrix;
        }

    template<class T2, int row2, int col2>
    Matrix<T, row, col2> operator*(Matrix<T2, row2, col2>& matrix2);

    T operator()(const int index1, const int index2);
    T* operator[](const int index);
    Matrix<T, row, col> operator-(Matrix<T, row, col>& matrix2);
    Matrix<T, row, col> operator+(Matrix<T, row, col>& matrix2);
    Matrix<T, row, col> operator*(T variable);
    friend Matrix<T, row, col> operator*(T variable, Matrix<T, row, col>& matrix2);
    void operator=(std::initializer_list<std::initializer_list<T>> list);
};

template<class T, int row, int col>
template<typename T2, int row2, int col2>
Matrix<T, row, col2> Matrix<T, row, col>::operator*(Matrix<T2, row2, col2>& matrix2)
{
    assert(col == row2 && "Col of first matrix dont equal row of second matrix");
    Matrix<T, row, col2> tempMatrix;
  for (int x = 0; x < row; ++x)
    for (int z = 0; z < col2; ++z) {
        tempMatrix.matrix[x][z] = 0;
    }
    for(int i = 0; i < row; ++i)
        for(int j = 0; j < col2; ++j)
            for(int l = 0; l < col; ++l)
            {
                tempMatrix.matrix[i][j] += matrix[i][l] * matrix2.matrix[l][j];
            }
    /*for(int n = 0; n < row; ++n)
        for(int m = 0; m < col2; ++m)
        {
            std::cout << tempMatrix.matrix[n][m] << std::endl;
        }
    */
    return tempMatrix;
}

template <class T, int row, int col>
T Matrix<T, row, col>::operator()(const int index1, const int index2)
{
    return matrix[index1][index2];
}

template <class T, int row, int col>
T* Matrix<T, row, col>::operator[](const int index)
{
    return matrix[index];
}

template<class T, int row, int col>
Matrix<T, row, col> Matrix<T, row, col>::operator-(Matrix<T, row, col>& matrix2)
{
    //assert(matrix2.matrix == col && matrix2.matrixRow == row && "Sizes of matrix dont equal");
    Matrix<T, row, col> tempMatrix;
    for(int i = 0; i < row; ++i)
        for(int j = 0; j < col; ++j)
            {
                tempMatrix[i][j] = matrix[i][j] - matrix2.matrix[i][j];
            }
    return tempMatrix;
}

template<class T, int row, int col>
Matrix<T, row, col> Matrix<T, row, col>::operator+(Matrix<T, row, col>& matrix2)
{
    //assert(matrix2.matrix == col && matrix2.matrixRow == row && "Sizes of matrix dont equal");
    Matrix<T, row, col> tempMatrix;
    for(int i = 0; i < row; ++i)
        for(int j = 0; j < col; ++j)
            {
                tempMatrix[i][j] = matrix[i][j] + matrix2.matrix[i][j];
            }
    return tempMatrix;
}

template<class T, int row, int col>
Matrix<T, row, col> Matrix<T, row, col>::operator*(T variable)
{
    Matrix<T, row, col> tempMatrix;
    for(int i = 0; i < row; ++i)
        for(int j = 0; j < col; ++j)
            {
                tempMatrix[i][j] = matrix[i][j] * variable;
            }
    return tempMatrix;
}

template<class T, int row, int col>
Matrix<T, row, col> operator*(T variable, Matrix<T, row, col>& matrix2)
{
    Matrix<T, row, col> tempMatrix;
    for(int i = 0; i < row; ++i)
        for(int j = 0; j < col; ++j)
            {
                tempMatrix[i][j] = matrix2[i][j] * variable;
            }

    return tempMatrix;
}

template <class T, int row, int col>
std::ostream& operator<<(std::ostream &ostream,
                               Matrix<T, row, col>& matrix)
{
    /*for (int (&array)[col] : matrix.matrix)
        for(int n : array)
            {
                ostream << n << " ";
            }*/
    for(int i = 0; i < row; ++i)
        for(int j = 0; j < col; ++j)
        {
            ostream << matrix.matrix[i][j] << " ";
        }

    return ostream;
}

template <typename X, int row, int col>
Matrix<X, 1, col> cutFunction(Matrix<X, row, col>& matrix, int iteration)
{
    Matrix<X, 1, col> tempMatrix;
    for(int i = 0; i < col; ++i)
    {
        tempMatrix[0][i] = matrix.matrix[iteration][i];
    }
    return tempMatrix;
}

template <class T, int row, int col>
void Matrix<T, row, col>::operator=(std::initializer_list<std::initializer_list<T>> list)
{
    typename std::initializer_list<std::initializer_list<T>>::const_iterator iterator;
    typename std::initializer_list<T>::const_iterator iterator2;
    iterator = list.begin();
    iterator2 = iterator->begin();
    for(int i = 0; i < row; ++i)
    {
        iterator2 = iterator->begin();
        ++iterator;
        for(int j = 0; j < col; ++j)
        {
            matrix[i][j] = *iterator2;
            ++iterator2;
        }
    }
}

/*template<class T, int row, int col>
void Matrix<T, row, col>::operator=(std::initializer_list<T> list)
{
    typename std::initializer_list<T>::const_iterator iterator;
    iterator = list.begin();
    for(int i = 0; i < row; ++i)
        for(int j = 0; j < col; ++j)
        {
            matrix[i][j] = *iterator;
            ++iterator;
        }
}*/

// Return X if X > 0; else return 0
template <class T, int row, int col>
void relu(Matrix<T, row, col>& matrix)
{
    for(int i = 0; i < row; ++i)
        for(int j = 0; j < col; ++j)
        {
            if(matrix[i][j] > 0)
                matrix[i][j] = matrix[i][j];
            else
                matrix[i][j] = 0;
        }
        //return (x > 0) * x;
}

// Return 1, if output > 0; else return 0
template<class T, int row, int col>
void relu2deriv(Matrix<T, row, col>& matrix, double* reluResult)
{

    for(int i = 0; i < row; ++i)
        for(int j = 0; j < col; ++j)
        {
            if(matrix[i][j] > 0)
                reluResult[j] = 1;
            else
                reluResult[j] = 0;
        }
}

double getRandomNumber(double min, double max)
{
    static const double fraction = 1.0 / (static_cast<double>(RAND_MAX) + 1.0);
    // Равномерно распределяем рандомное число в нашем диапазоне
    return static_cast<double>(rand() * fraction * (max - min + 1) + min)-1;
}

template <class T, int row, int col>
void weightsGenerator(Matrix<T, row, col>& matrix)
{
    for(int i = 0; i < row; ++i)
    {
        for(int j = 0; j < col; ++j)
        {
            matrix[i][j] = getRandomNumber(0.0, 1.0);
        }
    }
}
/*
void weightsGeneratorVector(double* weightsVector, int range)
{
    for(int i = 0; i < range; ++i)
    {
        weightsVector[i] = getRandomNumber(0.0, 1.0);
    }
}*/
