package com.ashutoshwad.utils.jautograd;

import java.util.Arrays;

class AbstractMatrix {
    private final double[][] value;
    private final double[][] gradient;
    protected final boolean requiresGradient;
    protected final ForwardComputeOperation forwardComputeOperation;
    protected final BackwardComputeOperation backwardComputeOperation;

    protected AbstractMatrix(double[][] value, double[][] gradient, boolean requiresGradient, ForwardComputeOperation forwardComputeOperation, BackwardComputeOperation backwardComputeOperation) {
        this.value = value;
        this.gradient = gradient;
        this.requiresGradient = requiresGradient;
        this.forwardComputeOperation = forwardComputeOperation;
        this.backwardComputeOperation = backwardComputeOperation;
    }

    // Accessor methods
    public void fill(double value) {
        for (int row = 0; row < numRows(); row++) {
            for (int col = 0; col < numCols(); col++) {
                setValue(row, col, value);
            }
        }
    }
    public double getValue() {
        return getValue(0, 0);
    }
    public double getGradient() {
        return getGradient(0, 0);
    }
    public double getValue(int row, int column) {
        return this.value[row][column];
    }
    public synchronized void setValue(int row, int column, double value) {
        this.value[row][column]=value;
    }
    public double getGradient(int row, int column) {
        return this.gradient[row][column];
    }
    public synchronized void setGradient(int row, int column, double value) {
        this.gradient[row][column]=value;
    }
    public synchronized void accumulateGradient(int row, int column, double value) {
        this.gradient[row][column]+=value;
    }
    public int numRows() {
        return this.value.length;
    }
    public int numCols() {
        return this.value[0].length;
    }
    public boolean getRequiresGradient() {
        return requiresGradient;
    }

    public String getPrintableMatrixValues() {
        StringBuilder sb = new StringBuilder();
        for (int row = 0; row < numRows(); row++) {
            for (int col = 0; col < numCols(); col++) {
                if (0==col) {
                    sb.append("| ");
                } else {
                    sb.append(", ");
                }
                sb.append(String.format("%3.7f", getValue(row, col)));
            }
            sb.append(" |\n");
        }
        return sb.toString();
    }

    public String getPrintableMatrixGradients() {
        StringBuilder sb = new StringBuilder();
        for (int row = 0; row < numRows(); row++) {
            for (int col = 0; col < numCols(); col++) {
                if (0==col) {
                    sb.append("| ");
                } else {
                    sb.append(", ");
                }
                sb.append(String.format("%3.7f", getGradient(row, col)));
            }
            sb.append(" |\n");
        }
        return sb.toString();
    }

    @Override
    public String toString() {
        return this.getClass().getSimpleName() + "{" +
                "value=["+numRows()+","+numCols()+"]" +
                (requiresGradient? ", gradient=["+numRows()+","+numCols()+"]":", gradient=[null]") +
                ", requiresGradient=" + requiresGradient +
                ", forwardComputeOperation=" + forwardComputeOperation +
                ", backwardComputeOperation=" + backwardComputeOperation +
                "}\n"+getPrintableMatrixValues();
    }
}
