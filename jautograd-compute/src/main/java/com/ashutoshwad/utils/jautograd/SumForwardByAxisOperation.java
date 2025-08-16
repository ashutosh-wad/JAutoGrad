package com.ashutoshwad.utils.jautograd;

class SumForwardByAxisOperation extends ForwardComputeOperation {
    private final Matrix source;
    private final int axis;
    public SumForwardByAxisOperation(Matrix source, int axis, ForwardComputeOperation... forwardComputeOperations) {
        super(null, null, forwardComputeOperations);
        this.source = source;
        this.axis = axis;
    }

    @Override
    protected void perform() {
        if(axis == 0) {
            sumAcrossRows();
        } else {
            sumAcrossCols();
        }
    }

    private void sumAcrossRows() {
        Matrix result = getResult();
        for (int col = 0; col < result.numCols(); col++) {
            double temp = 0;
            for (int row = 0; row < source.numRows(); row++) {
                temp += source.getValue(row, col);
            }
            result.setValue(0, col, temp);
        }
    }

    private void sumAcrossCols() {
        Matrix result = getResult();
        for (int row = 0; row < result.numRows(); row++) {
            double temp = 0;
            for (int col = 0; col < source.numCols(); col++) {
                temp += source.getValue(row, col);
            }
            result.setValue(row, 0, temp);
        }
    }
}
