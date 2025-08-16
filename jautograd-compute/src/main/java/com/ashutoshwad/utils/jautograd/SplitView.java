package com.ashutoshwad.utils.jautograd;

public class SplitView extends Matrix {
    private final Matrix original;
    private final int rowStart;
    private final int rowLength;
    private final int colStart;
    private final int colLength;

    public SplitView(Matrix original, int rowStart, int rowLength, int colStart, int colLength) {
        super(original.requiresGradient, original.forwardComputeOperation, original.backwardComputeOperation);
        this.original = original;
        this.rowStart = rowStart;
        this.rowLength = rowLength;
        this.colStart = colStart;
        this.colLength = colLength;
    }

    private int mapRow(int index) {
        if (index >= rowLength || index < 0) {
            throw new ArrayIndexOutOfBoundsException("This matrix has " + rowLength + " rows. An attempt was made to access row: " + index);
        }
        return index + rowStart;
    }

    private int mapCol(int index) {
        if (index >= colLength || index < 0) {
            throw new ArrayIndexOutOfBoundsException("This matrix has " + colLength + " columns. An attempt was made to access column: " + index);
        }
        return index + colStart;
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
        return rowLength;
    }
    public int numCols() {
        return colLength;
    }
}
