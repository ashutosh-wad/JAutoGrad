package com.ashutoshwad.utils.jautograd;

import java.util.Arrays;

public class MatrixStore {
    private float[] data;
    private int numRows;
    private int numCols;

    public MatrixStore(int numRows, int numCols) {
        this(numRows, numCols, null);
    }

    public MatrixStore(int numRows, int numCols, float[]backingArray) {
        if (numRows<=0) {
            throw new IllegalArgumentException("A matrix must have at least 1 row");
        }
        if (numCols<=0) {
            throw new IllegalArgumentException("A matrix must have at least 1 column");
        }
        if (null == backingArray) {
            backingArray = new float[numRows * numCols];
        }
        if (backingArray.length == 0) {
            throw new IllegalArgumentException("Backing array cannot be empty.");
        }
        if (backingArray.length < (numRows * numCols)) {
            throw new IllegalArgumentException("Backing array is not big enough.");
        }
        this.numRows = numRows;
        this.numCols = numCols;
        this.data = backingArray;
    }

    public int numRows() {
        return numRows;
    }

    public int numCols() {
        return numCols;
    }

    /**
     * Retrieve a single element in the matrix
     *
     * @param row    The row index of the element
     * @param column The column index of the element
     * @return The value present at (row, column)
     */
    public float get(int row, int column) {
        return data[row * numCols  + column];
    }

    /**
     * Set the value of a single element in the matrix.
     *
     * @param row    The row index of the element
     * @param column The column index of the element
     * @param value  The value to set to the element of the matrix
     */
    public void set(int row, int column, float value) {
        data[row * numCols  + column] = value;
    }

    /**
     * Add the provided value to a single element in the matrix.
     *
     * @param row    The row index of the element
     * @param column The column index of the element
     * @param value  The value to be added to the element of the matrix
     */
    public void add(int row, int column, float value) {
        data[row * numCols  + column] += value;
    }

    /**
     * Transposes the matrix in place
     */
    public void transpose() {
        if (numRows == numCols) {
            float temp = 0.0f;
            for (int row = 0; row < numRows; row++) {
                for (int col = 0; col < numCols; col++) {
                    if (row == col) {
                        continue;
                    }
                    int oIndex = row * numCols + col;
                    int nIndex = col * numRows + row;
                    temp = data[oIndex];
                    data[oIndex] = data[nIndex];
                    data[nIndex] = temp;
                }
            }
        } else {
            float[]newData = new float[data.length];

            for (int row = 0; row < numRows; row++) {
                for (int col = 0; col < numCols; col++) {
                    if (row == col) {
                        continue;
                    }
                    int oIndex = row * numCols + col;
                    int nIndex = col * numRows + row;
                    newData[nIndex] = data[oIndex];
                }
            }

            data = newData;
        }

        int tmp = numCols;
        numCols = numRows;
        numRows = tmp;
    }

    public void fill(float value) {
        Arrays.fill(data, value);
    }

    /**
     * Returns the backing array of this matrix in case you wanted to do some fancy in memory stuff.
     *
     * @return
     */
    public float[] backingArray() {
        return data;
    }

    /**
     * Translates the given row and column to an index into the backing array.
     *
     * @param row    The row index of the element in the current matrix.
     * @param column The column index of the element in the matrix.
     * @return The index of the element in the backing array.
     */
    public int computeIndex(int row, int column) {
        return row * numCols  + column;
    }
}
