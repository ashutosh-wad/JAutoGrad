package com.ashutoshwad.utils.jautograd.transformer;

import com.ashutoshwad.utils.jautograd.Matrix;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

public class TransformerPersistantUtils {
    public static void readMatrix(Matrix mat, ObjectInputStream is) {
        try {
            for (int row = 0; row < mat.numRows(); row++) {
                for (int col = 0; col < mat.numCols(); col++) {
                    mat.setValue(row, col, is.readDouble());
                }
            }
        } catch (Exception e) {
            throw new RuntimeException("Mayday in the boogie!", e);
        }
    }

    public static void writeMatrix(Matrix mat, ObjectOutputStream os) {
        try {
            for (int row = 0; row < mat.numRows(); row++) {
                for (int col = 0; col < mat.numCols(); col++) {
                    os.writeDouble(mat.getValue(row, col));
                }
            }
        } catch (Exception e) {
            throw new RuntimeException("Mayday in the boogie!", e);
        }
    }
}
