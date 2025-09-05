package com.ashutoshwad.utils.jautograd;

public class MatrixStorage {
    private final float[] data;
    private final int numRows;
    private final int numCols;

    public MatrixStorage(int numRows, int numCols) {
        this.numRows = numRows;
        this.numCols = numCols;
        this.data = new float[numRows * numCols];
    }

    public float get(int row, int col) {
        return data[row * numCols + col];
    }

    public void set(int row, int col, float value) {
        data[row * numCols + col] = value;
    }

    public void accumulate(int row, int col, float value) {
        data[row * numCols + col] += value;
    }

    public int getNumRows() {
        return numRows;
    }

    public int getNumCols() {
        return numCols;
    }
}
