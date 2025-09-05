package com.ashutoshwad.utils.jautograd;

class AbstractMatrix {
    private final MatrixStorage value;
    private final MatrixStorage gradient;
    protected final boolean requiresGradient;
    protected final ForwardComputeOperation forwardComputeOperation;
    protected final BackwardComputeOperation backwardComputeOperation;

    protected AbstractMatrix(MatrixStorage value, MatrixStorage gradient, boolean requiresGradient, ForwardComputeOperation forwardComputeOperation, BackwardComputeOperation backwardComputeOperation) {
        this.value = value;
        this.gradient = gradient;
        this.requiresGradient = requiresGradient;
        this.forwardComputeOperation = forwardComputeOperation;
        this.backwardComputeOperation = backwardComputeOperation;
    }

    // Accessor methods
    public void fill(float value) {
        for (int row = 0; row < numRows(); row++) {
            for (int col = 0; col < numCols(); col++) {
                setValue(row, col, value);
            }
        }
    }
    public float getValue() {
        return getValue(0, 0);
    }
    public float getGradient() {
        return getGradient(0, 0);
    }
    public float getValue(int row, int column) {
        return this.value.get(row, column);
    }
    public synchronized void setValue(int row, int column, float value) {
        this.value.set(row, column, value);
    }
    public float getGradient(int row, int column) {
        return this.gradient.get(row, column);
    }
    public synchronized void setGradient(int row, int column, float value) {
        this.gradient.set(row, column, value);
    }
    public synchronized void accumulateGradient(int row, int column, float value) {
        this.gradient.accumulate(row, column, value);
    }
    public int numRows() {
        return this.value.getNumRows();
    }
    public int numCols() {
        return this.value.getNumCols();
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
