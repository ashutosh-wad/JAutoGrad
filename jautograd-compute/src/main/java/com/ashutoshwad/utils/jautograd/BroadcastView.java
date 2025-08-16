package com.ashutoshwad.utils.jautograd;

class BroadcastView extends Matrix {
    private static final String ERROR_MESSAGE = "Matrix (%d, %d) is not broadcastable to (%d, %d)";
    private final Matrix original;
    private final int bRows;
    private final int bCols;
    private BroadcastView(Matrix original, int rows, int cols) {
        super(original.requiresGradient, original.forwardComputeOperation, original.backwardComputeOperation);
        this.original = original;
        bRows = rows;
        bCols = cols;
    }

    public static Matrix broadcast(Matrix m, int rows, int cols) {
        if(m.numRows() != rows && m.numRows()!=1) {
            throw new IllegalArgumentException(String.format(ERROR_MESSAGE, m.numRows(), m.numCols(), rows, cols));
        }
        if(m.numCols() != cols && m.numCols()!=1) {
            throw new IllegalArgumentException(String.format(ERROR_MESSAGE, m.numRows(), m.numCols(), rows, cols));
        }
        if(m.numRows() == rows && m.numCols() == cols) {
            return m;//No need to brodcast
        }
        return new BroadcastView(m, rows, cols);
    }

    private int mapRow(int row) {
        if (row >= bRows || row < 0) {
            throw new ArrayIndexOutOfBoundsException("This matrix has " + bRows + " rows. An attempt was made to access row: " + row);
        }
        return (original.numRows() == 1) ? 0 : row;
    }

    private int mapCol(int column) {
        if (column >= bCols || column < 0) {
            throw new ArrayIndexOutOfBoundsException("This matrix has " + bCols + " columns. An attempt was made to access column: " + column);
        }
        return (original.numCols() == 1) ? 0 : column;
    }

    //Override accessors
    public double getValue(int row, int column) {
        return original.getValue(mapRow(row), mapCol(column));
    }
    public synchronized void setValue(int row, int column, double value) {
        original.setValue(mapRow(row), mapCol(column), value);
    }
    public double getGradient(int row, int column) {
        return original.getGradient(mapRow(row), mapCol(column));
    }
    public synchronized void setGradient(int row, int column, double value) {
        original.setGradient(mapRow(row), mapCol(column), value);
    }
    public synchronized void accumulateGradient(int row, int column, double value) {
        original.accumulateGradient(mapRow(row), mapCol(column), value);
    }
    public int numRows() {
        return bRows;
    }
    public int numCols() {
        return bCols;
    }
}
