package com.ashutoshwad.utils.jautograd;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

public class BinaryFunctionTest {
    @Test
    public void brodcastTest() {
        Matrix m1 = new Matrix(3, 3, true);
        m1.fillValues(2);

        Matrix m2 = new Matrix(3, 3, true);
        m2.fillValues(3);

        Matrix m3 = m1.mul(m2);
        m3.backward();

        assertEquals(3, m3.numRows());
        assertEquals(3, m3.numCols());
        for (int row = 0; row < m3.numRows(); row++) {
            for (int col = 0; col < m3.numCols(); col++) {
                assertEquals(6, m3.getValue(row, col));
            }
        }

        //Now check row broadcast
        m1 = new Matrix(3, 3, true);
        m1.fillValues(2);

        m2 = new Matrix(1, 3, true);
        m2.fillValues(3);

        m3 = m1.mul(m2);
        m3.backward();
        assertEquals(6, m2.getGradient(0, 0));

        assertEquals(3, m3.numRows());
        assertEquals(3, m3.numCols());
        for (int row = 0; row < m3.numRows(); row++) {
            for (int col = 0; col < m3.numCols(); col++) {
                assertEquals(6, m3.getValue(row, col));
            }
        }

        //Now check column broadcast
        m1 = new Matrix(3, 3, true);
        m1.fillValues(2);

        m2 = new Matrix(3, 1, true);
        m2.fillValues(3);

        m3 = m1.mul(m2);
        m3.backward();
        assertEquals(6, m2.getGradient(0, 0));

        assertEquals(3, m3.numRows());
        assertEquals(3, m3.numCols());
        for (int row = 0; row < m3.numRows(); row++) {
            for (int col = 0; col < m3.numCols(); col++) {
                assertEquals(6, m3.getValue(row, col));
            }
        }

        //Now check scalar broadcast
        m1 = new Matrix(3, 3, true);
        m1.fillValues(2);

        m2 = new Matrix(1, 1, true);
        m2.fillValues(3);

        m3 = m1.mul(m2).add(new Matrix(1, 1));
        m3.zeroGradAndForward();
        m3.backward();
        assertEquals(18, m2.getGradient(0, 0));

        assertEquals(3, m3.numRows());
        assertEquals(3, m3.numCols());
        for (int row = 0; row < m3.numRows(); row++) {
            for (int col = 0; col < m3.numCols(); col++) {
                assertEquals(6, m3.getValue(row, col));
            }
        }

    }
}
