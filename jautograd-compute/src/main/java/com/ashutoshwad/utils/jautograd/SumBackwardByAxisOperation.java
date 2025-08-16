package com.ashutoshwad.utils.jautograd;

public class SumBackwardByAxisOperation extends BackwardComputeOperation {
    private final Matrix source;
    private final int axis;
    public SumBackwardByAxisOperation(Matrix source, int axis, BackwardComputeOperation... backwardComputeOperations) {
        super(null, null, backwardComputeOperations);
        this.source = source;
        this.axis = axis;
    }

    @Override
    protected void perform() {
        if(axis == 0) {
            sumBackwardAcrossRows();
        } else {
            sumBackwardAcrossCols();
        }
    }

    private void sumBackwardAcrossRows() {
        Matrix result = getResult();
        for (int col = 0; col < result.numCols(); col++) {
            double gradient = result.getGradient(0, col);
            for (int row = 0; row < source.numRows(); row++) {
                source.accumulateGradient(row, col, gradient);
            }
        }
    }

    private void sumBackwardAcrossCols() {
        Matrix result = getResult();
        for (int row = 0; row < result.numRows(); row++) {
            double gradient = result.getGradient(row, 0);
            for (int col = 0; col < source.numCols(); col++) {
                source.accumulateGradient(row, col, gradient);
            }
        }
    }
}
