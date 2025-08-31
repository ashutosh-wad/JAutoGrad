package com.ashutoshwad.utils.jautograd.transformer;

import com.ashutoshwad.utils.jautograd.Matrix;

public class RotaryPositionEncoder {
    private final int numRows;
    private final int numCols;
    private final Matrix oddColumnExtractor;
    private final Matrix evenColumnExtractor;
    private final Matrix oddColumnSpreader;
    private final Matrix evenColumnSpreader;
    private final Matrix sinMatrix;
    private final Matrix cosMatrix;

    public RotaryPositionEncoder(int numRows, int numCols) {
        if (numCols % 2 != 0) {
            throw new IllegalArgumentException("For RoPE number of columns must be even. Number of columns provided was: " + numCols);
        }
        this.numRows = numRows;
        this.numCols = numCols;
        final int numColsByTwo = numCols / 2;
        this.oddColumnExtractor = createOddColumnExtractorMatrix(numCols);
        this.evenColumnExtractor = createEvenColumnExtractorMatrix(numCols);
        this.oddColumnSpreader = oddColumnExtractor.transpose();
        this.evenColumnSpreader = evenColumnExtractor.transpose();
        this.sinMatrix = createSinMatrix(numRows, numColsByTwo);
        this.cosMatrix = createCosMatrix(numRows, numColsByTwo);
    }

    public Matrix rotate(Matrix source) {
        if(source.numRows()!=numRows || source.numCols()!=numCols) {
            throw new IllegalArgumentException("Expected a matrix of size ["+numRows+", "+numCols+"]. Found ["+source.numRows()+", "+source.numCols()+"]");
        }
        Matrix a = source.matmul(oddColumnExtractor);
        Matrix b = source.matmul(evenColumnExtractor);
        Matrix aCosTheta = a.mul(cosMatrix);
        Matrix bCosTheta = b.mul(cosMatrix);
        Matrix aSinTheta = a.mul(sinMatrix);
        Matrix bSinTheta = b.mul(sinMatrix);
        Matrix oddRows = aCosTheta.add(bSinTheta);
        Matrix evenRows = bCosTheta.sub(aSinTheta);
        Matrix oddSpread = oddRows.matmul(oddColumnSpreader);
        Matrix evenSpread = evenRows.matmul(evenColumnSpreader);
        return oddSpread.add(evenSpread);
    }

    public static double calculateTheta(double i, double m, double dk) {
        return Math.pow((1.0/10000), i / dk) * m;
    }

    public static Matrix createSinMatrix(final int numRows, final int numCols) {
        Matrix sinMatrix = Matrix.create(numRows, numCols);
        for (int row = 0; row < sinMatrix.numRows(); row++) {
            for (int col = 0; col < sinMatrix.numCols(); col++) {
                sinMatrix.setValue(row, col, Math.sin(calculateTheta(col, row, numCols)));
            }
        }
        return sinMatrix;
    }

    public static Matrix createCosMatrix(final int numRows, final int numCols) {
        Matrix cosMatrix = Matrix.create(numRows, numCols);
        for (int row = 0; row < cosMatrix.numRows(); row++) {
            for (int col = 0; col < cosMatrix.numCols(); col++) {
                cosMatrix.setValue(row, col, Math.cos(calculateTheta(col, row, numCols)));
            }
        }
        return cosMatrix;
    }

    public static Matrix createOddColumnExtractorMatrix(final int numColumns) {
        int trueNumCols = numColumns / 2 + numColumns % 2;
        Matrix extractor = Matrix.create(numColumns, trueNumCols);
        int row = 0;
        int col = 0;
        while (row < extractor.numRows() && col < extractor.numCols()) {
            extractor.setValue(row, col, 1);
            row+=2;
            col+=1;
        }
        return extractor;
    }

    public static Matrix createEvenColumnExtractorMatrix(final int numColumns) {
        int trueNumCols = numColumns / 2;
        Matrix extractor = Matrix.create(numColumns, trueNumCols);
        int row = 1;
        int col = 0;
        while (row < extractor.numRows() && col < extractor.numCols()) {
            extractor.setValue(row, col, 1);
            row+=2;
            col+=1;
        }
        return extractor;
    }
}
