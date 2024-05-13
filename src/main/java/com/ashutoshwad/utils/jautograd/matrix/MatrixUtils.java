package com.ashutoshwad.utils.jautograd.matrix;

import java.util.function.DoubleSupplier;
import java.util.function.Function;

import com.ashutoshwad.utils.jautograd.Value;

public class MatrixUtils {
	/**
	 * This function multiplies the matrices given in the input and returns the
	 * result.
	 * 
	 * @param a First matrix
	 * @param b Second matrix
	 * @return result of multiplying a and b
	 */
	public static Value[][] mul(Value[][] a, Value[][] b) {
		int rowsA = a.length;
		int colsA = a[0].length;
		int rowsB = b.length;
		int colsB = b[0].length;

		if (colsA != rowsB)
			throw new RuntimeException("Matrix dimensions don't match!");

		Value[][] result = new Value[rowsA][colsB];

		for (int i = 0; i < rowsA; i++) {
			for (int j = 0; j < colsB; j++) {
				Value sum = Value.of(0);
				for (int k = 0; k < colsA; k++) {
					sum = sum.add(a[i][k].mul(b[k][j]));
				}
				result[i][j] = sum;
			}
		}

		return result;
	}

	/**
	 * This method returns the transpose of the matrix provided in the input.
	 * 
	 * @param matrix to be transposed
	 * @return the transposed matrix
	 */
	public static Value[][] transpose(Value[][] matrix) {
		int rows = matrix.length;
		int cols = matrix[0].length;
		Value[][] transposed = new Value[cols][rows];

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				transposed[j][i] = matrix[i][j];
			}
		}
		return transposed;
	}

	/**
	 * This method accepts as input a Value matrix. For each value in the matrix it
	 * then performs the provided operation.
	 * 
	 * @param matrix    the matrix to be modified
	 * @param operation the operation to be used to modify all the values in the
	 *                  matrix
	 * @return the modified matrix
	 */
	public static Value[][] op(Value[][] matrix, Function<Value, Value> operation) {
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[i].length; j++) {
				matrix[i][j] = operation.apply(matrix[i][j]);
			}
		}
		return matrix;
	}

	/**
	 * Create a matrix of the specified dimensions where each value is initialized
	 * to zero
	 * 
	 * @param numRows the number of rows that the matrix should have
	 * @param numCols the number of columns that the matrix should have
	 * @return the newly created matrix
	 */
	public static Value[][] createMatrix(int numRows, int numCols) {
		return createMatrix(numRows, numCols, () -> 0.0);
	}

	/**
	 * Create a matrix of the specified dimensions where each value is initialized
	 * by the supplier specified
	 * 
	 * @param numRows the number of rows that the matrix should have
	 * @param numCols the number of columns that the matrix should have
	 * @return the newly created matrix
	 */
	public static Value[][] createMatrix(int numRows, int numCols, DoubleSupplier supplier) {
		Value[][] retVal = new Value[numRows][numCols];
		for (int i = 0; i < numRows; i++) {
			for (int j = 0; j < numCols; j++) {
				retVal[i][j] = Value.learnable(supplier.getAsDouble());
			}
		}
		return retVal;
	}

	/**
	 * This method prints out the matrix, it is useful when debugging issues.
	 * @param output the matrix to print
	 */
	public static void print(Value[][] output) {
		for (int i = 0; i < output.length; i++) {
			for (int j = 0; j < output[i].length; j++) {
				System.out.print(output[i][j] + "\t");
			}
			System.out.println();
		}
	}
}
