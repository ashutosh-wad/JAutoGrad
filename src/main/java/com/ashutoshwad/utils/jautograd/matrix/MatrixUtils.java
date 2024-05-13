package com.ashutoshwad.utils.jautograd.matrix;

import com.ashutoshwad.utils.jautograd.Value;

public class MatrixUtils {
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
}
