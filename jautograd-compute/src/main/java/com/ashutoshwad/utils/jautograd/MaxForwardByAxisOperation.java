package com.ashutoshwad.utils.jautograd;

class MaxForwardByAxisOperation extends ForwardComputeOperation {
    private final Matrix source;
    private final int axis;
    public MaxForwardByAxisOperation(Matrix source, int axis, ForwardComputeOperation... forwardComputeOperations) {
        super(null, null, forwardComputeOperations);
        this.source = source;
        this.axis = axis;
    }

    @Override
    protected void perform() {
        if(axis == 0) {
            maxAcrossRows();
        } else {
            maxAcrossCols();
        }
    }

    private void maxAcrossRows() {
        Matrix result = getResult();
        for (int col = 0; col < result.numCols(); col++) {
            double max = 0;
            for (int row = 0; row < source.numRows(); row++) {
                double sourceVal = source.getValue(row, col);
                if(row == 0) {
                    max = sourceVal;
                } else {
                    max = Math.max(max, sourceVal);
                }
            }
            result.setValue(0, col, max);
        }
    }

    private void maxAcrossCols() {
        Matrix result = getResult();
        for (int row = 0; row < result.numRows(); row++) {
            double max = 0;
            for (int col = 0; col < source.numCols(); col++) {
                double sourceVal = source.getValue(row, col);
                if(col == 0) {
                    max = sourceVal;
                } else {
                    max = Math.max(max, sourceVal);
                }
            }
            result.setValue(row, 0, max);
        }
    }
}
