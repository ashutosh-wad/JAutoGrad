package com.ashutoshwad.utils.jautograd;

class MinForwardByAxisOperation extends ForwardComputeOperation {
    private final Matrix source;
    private final int axis;
    public MinForwardByAxisOperation(Matrix source, int axis, ForwardComputeOperation... forwardComputeOperations) {
        super(null, null, forwardComputeOperations);
        this.source = source;
        this.axis = axis;
    }

    @Override
    protected void perform() {
        if(axis == 0) {
            minAcrossRows();
        } else {
            minAcrossCols();
        }
    }

    private void minAcrossRows() {
        Matrix result = getResult();
        for (int col = 0; col < result.numCols(); col++) {
            double min = 0;
            for (int row = 0; row < source.numRows(); row++) {
                double sourceVal = source.getValue(row, col);
                if(row == 0) {
                    min = sourceVal;
                } else {
                    min = Math.min(min, sourceVal);
                }
            }
            result.setValue(0, col, min);
        }
    }

    private void minAcrossCols() {
        Matrix result = getResult();
        for (int row = 0; row < result.numRows(); row++) {
            double min = 0;
            for (int col = 0; col < source.numCols(); col++) {
                double sourceVal = source.getValue(row, col);
                if(col == 0) {
                    min = sourceVal;
                } else {
                    min = Math.min(min, sourceVal);
                }
            }
            result.setValue(row, 0, min);
        }
    }
}
