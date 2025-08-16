package com.ashutoshwad.utils.jautograd;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class MatrixTest {
    @Test
    public void testIdentityMultiplication() {
        Matrix a = Matrix.create(3, 3, ()->3.0, true);
        Matrix b = Matrix.create(3, 3, true);
        b.setValue(0, 0, 1);
        b.setValue(1, 1, 1);
        b.setValue(2, 2, 1);

        Matrix c = a.matmul(b).add(Matrix.create(0.0)).relu();
        c.backward();
        for (int row = 0; row < c.numRows(); row++) {
            for (int col = 0; col < c.numCols(); col++) {
                assertEquals(3, c.getValue(row, col));
                assertEquals(1, c.getGradient(row, col));
                assertEquals(1, a.getGradient(row, col));
                assertEquals(9, b.getGradient(row, col));
            }
        }
    }
}
