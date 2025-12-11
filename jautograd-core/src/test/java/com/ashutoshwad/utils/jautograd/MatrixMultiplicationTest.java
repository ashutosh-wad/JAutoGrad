package com.ashutoshwad.utils.jautograd;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class MatrixMultiplicationTest {
    @Test
    public void identity_matrix_multiplication_should_work() {
        int rows = 1024;
        int cols = 1024;
        Matrix m1 = Matrix.createXavierGlorotInitializedMatrix(rows, cols);
        Matrix m2 = new Matrix(rows, cols);
        for (int row = 0, col = 0; row < rows; row++, col++) {
            m2.setValue(row, col, 1);
        }

        Matrix m3 = m1.matmul(m2);

        long start = 0;
        long end = 0;

        start = System.currentTimeMillis();
        m3.forward();
        end = System.currentTimeMillis();
        System.out.println(end-start);

        for (int row = 0; row < m1.numRows(); row++) {
            for (int col = 0; col < m1.numCols(); col++) {
                assertEquals(m1.getValue(row, col), m3.getValue(row, col));
            }
        }
    }
}
