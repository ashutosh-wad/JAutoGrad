package com.ashutoshwad.utils.jautograd.transformer;

import static org.junit.jupiter.api.Assertions.*;
import com.ashutoshwad.utils.jautograd.Matrix;
import org.junit.jupiter.api.Test;

public class RotaryPositionEncoderTest {
    @Test
    public void testRopeWorks() {
        Matrix matty = Matrix.create(2, 8, ()->0.2);
        for (int row = 0; row < matty.numRows(); row++) {
            double start = 0.1;
            for (int col = 0; col < matty.numCols(); col++) {
                matty.setValue(row, col, start);
                start+=0.1;
            }
        }
        RotaryPositionEncoder enc = new RotaryPositionEncoder(2, 8);
        Matrix rotated = enc.rotate(matty);
        System.out.println();
        double a = 0.1000000;
        System.out.println("a: " + a);
        double b = 0.2000000;
        System.out.println("b: " + b);
        double aCosTheta = a * Math.cos(1);
        System.out.println("aCosTheta: " + aCosTheta);
        double bCosTheta = b * Math.cos(1);
        System.out.println("bCosTheta: " + bCosTheta);
        double aSinTheta = a * Math.sin(1);
        System.out.println("aSinTheta: " + aSinTheta);
        double bSinTheta = b * Math.sin(1);
        System.out.println("bSinTheta: " + bSinTheta);
        System.out.println("aCosTheta + bSinTheta: " + (aCosTheta + bSinTheta));
        assertEquals((aCosTheta + bSinTheta),rotated.getValue(1, 0), 1e-9);
        System.out.println("bCosTheta - aSinTheta: " + (bCosTheta - aSinTheta));
        assertEquals((bCosTheta - aSinTheta),rotated.getValue(1, 1), 1e-9);
    }
}
