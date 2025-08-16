package com.ashutoshwad.utils.jautograd;

public class TransposeView extends Matrix {
    private Matrix original;
    public TransposeView(Matrix original) {
        super(original.requiresGradient, original.forwardComputeOperation, original.backwardComputeOperation);
        this.original = original;
    }

    //Override accessors
    public double getValue(int row, int column) {
        return original.getValue(column, row);
    }
    public synchronized void setValue(int row, int column, double value) {
        original.setValue(column, row, value);
    }
    public double getGradient(int row, int column) {
        return original.getGradient(column, row);
    }
    public synchronized void setGradient(int row, int column, double value) {
        original.setGradient(column, row, value);
    }
    public synchronized void accumulateGradient(int row, int column, double value) {
        original.accumulateGradient(column, row, value);
    }
    public int numRows() {
        return original.numCols();
    }
    public int numCols() {
        return original.numRows();
    }
}
