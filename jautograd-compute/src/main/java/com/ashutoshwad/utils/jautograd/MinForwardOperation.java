package com.ashutoshwad.utils.jautograd;

class MinForwardOperation extends ForwardComputeOperation {
    private final Matrix source;
    public MinForwardOperation(Matrix source, ForwardComputeOperation... forwardComputeOperations) {
        super(null, null, forwardComputeOperations);
        this.source = source;
    }

    @Override
    protected void perform() {
        Matrix result = getResult();
        double min = 0;
        for (int row = 0; row < source.numRows(); row++) {
            for (int col = 0; col < source.numCols(); col++) {
                double sourceVal = source.getValue(row, col);
                if(row == 0 && col == 0) {
                    min = sourceVal;
                } else {
                    min = Math.min(min, sourceVal);
                }
            }
        }
        result.setValue(0, 0, min);
    }
}
