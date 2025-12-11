package com.ashutoshwad.utils.jautograd;

/**
 * A lightweight matrix view over a 1D float array with given base and strides.
 * <p>
 * Memory layout by default is row-major (rowStride = numCols, columnStride = 1).
 * Views and transposes share the same backing array (zero-copy).
 * <b>Not thread-safe.</b> Writes in any view affect all aliasing views.
 */
public class MatrixStoreOld {
    private final float[] data;
    private final int base;
    private final int numRows;
    private final int numCols;
    private final int rowStride;
    private final int columnStride;

    /**
     * Create a new, zero-initialized matrix with the given dimensions.
     *
     * @param rows number of rows
     * @param columns number of columns
     * @throws IllegalArgumentException if rows or cols are negative
     */
    public MatrixStoreOld(int rows, int columns) {
        matrixMustContainAtLeastOneElement(rows, columns);
        this.data = new float[rows * columns];
        this.base = 0;
        this.numRows = rows;
        this.numCols = columns;
        this.rowStride = columns;
        this.columnStride = 1;
    }

    /**
     * An internal constructor used to create views to this same backing storage.
     *
     * @param data         The backing array
     * @param base         The base i.e. first index at 0, 0
     * @param numRows      number of rows that the matrix should have
     * @param numCols      number of columns that the matrix should have
     * @param rowStride    when added to row index gives the next row
     * @param columnStride when added to column index gives the next column
     */
    private MatrixStoreOld(float[] data, int base, int numRows, int numCols, int rowStride, int columnStride) {
        if(rowStride<=0) {
            throw new IllegalArgumentException("row stride cannot be 0 or negative!");
        }
        if(columnStride<=0) {
            throw new IllegalArgumentException("column stride cannot be 0 or negative!");
        }
        this.data = data;
        this.base = base;
        this.numRows = numRows;
        this.numCols = numCols;
        this.rowStride = rowStride;
        this.columnStride = columnStride;
    }

    private void matrixMustContainAtLeastOneElement(int rows, int columns) {
        if (rows <= 0) {
            throw new IllegalArgumentException("A matrix must have at least one row. You have requested for: " + rows);
        }
        if (columns <= 0) {
            throw new IllegalArgumentException("A matrix must have at least one column. You have requested for: " + columns);
        }
    }

    /**
     * Retrieve a single element in the matrix
     *
     * @param row    The row index of the element
     * @param column The column index of the element
     * @return The value present at (row, column)
     */
    public float get(int row, int column) {
        return data[base + row * rowStride + column * columnStride];
    }

    /**
     * Set the value of a single element in the matrix.
     *
     * @param row    The row index of the element
     * @param column The column index of the element
     * @param value  The value to set to the element of the matrix
     */
    public void set(int row, int column, float value) {
        data[base + row * rowStride + column * columnStride] = value;
    }

    /**
     * Add the provided value to a single element in the matrix.
     *
     * @param row    The row index of the element
     * @param column The column index of the element
     * @param value  The value to be added to the element of the matrix
     */
    public void add(int row, int column, float value) {
        data[base + row * rowStride + column * columnStride] += value;
    }

    /**
     * Create a subview of the matrix so that memory is not duplicated. The resultant matrix will behave no differently to an actual matrix.
     *
     * @param row     The row index of the first element in the new matrix
     * @param column  The column index of the first element in the new matrix
     * @param rows    The height of the new matrix, we can also call this number of rows in the new matrix.
     * @param columns The width of the new matrix, we can also call this number of columns in the new matrix.
     * @return The new view to the current matrix with the specified dimensions.
     */
    public MatrixStoreOld subview(int row, int column, int rows, int columns) {
        matrixMustContainAtLeastOneElement(rows, columns);
        int updatedBase = base + row * rowStride + column * columnStride;
        int largestElement = updatedBase + (rows - 1) * rowStride + (columns - 1) * columnStride;
        if (largestElement >= data.length) {
            throw new IllegalArgumentException("This matrix ("+rows+", "+columns+") cannot fit in the backing array starting at "+updatedBase+", it will overflow!");
        }
        if (row < 0 || column < 0) {
            throw new IllegalArgumentException("Row and column must be positive. Received ("+row+", "+column+").");
        }
        if ((rows > numRows() - row) || (columns > numCols() - column)) {
            throw new IllegalArgumentException("This matrix ("+rows+", "+columns+") cannot fit in this matrix ("+numRows()+", "+numCols()+") starting at index ("+row+", "+column+"), it will overflow!");
        }
        return new MatrixStoreOld(data, updatedBase, rows, columns, rowStride, columnStride);
    }

    /**
     * Returns a fast transpose of the current matrix
     *
     * @return a fast transpose of the current matrix
     */
    public MatrixStoreOld transpose() {
        return new MatrixStoreOld(data, base, numCols, numRows, columnStride, rowStride);
    }

    /**
     * The base index of the matrix in the backing array. This can also be thought of as the index in the backing array corresponding to the element (0, 0)
     *
     * @return The base index of the matrix in the backing array. This can also be thought of as the index in the backing array corresponding to the element (0, 0)
     */
    public int base() {
        return base;
    }

    /**
     * This is the amount that when added to the current row index, will give the index of the next row in the backing array.
     *
     * @return
     */
    public int rowStride() {
        return rowStride;
    }

    /**
     * This is the amount that when added to the current column index, will give the index of the next column in the backing array.
     *
     * @return
     */
    public int columnStride() {
        return columnStride;
    }

    /**
     * Number of rows present in this Matrix
     *
     * @return
     */
    public int numRows() {
        return numRows;
    }

    /**
     * Number of columns present in this Matrix
     *
     * @return
     */
    public int numCols() {
        return numCols;
    }

    /**
     * Set all the elements of this matrix equal to the provided value.
     *
     * @param value The value to set all elements of this matrix.
     */
    public void fill(float value) {
        for (int row = 0; row < numRows; row++) {
            int offset = base + row * rowStride;
            for (int col = 0; col < numCols; col++) {
                data[offset + col * columnStride] = value;
            }
        }
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
        return base + row * rowStride + column * columnStride;
    }
}
