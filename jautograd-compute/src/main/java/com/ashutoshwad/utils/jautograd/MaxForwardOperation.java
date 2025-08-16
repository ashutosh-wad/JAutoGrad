package com.ashutoshwad.utils.jautograd;

class MaxForwardOperation extends ForwardComputeOperation {
    private final Matrix source;
    public MaxForwardOperation(Matrix source, ForwardComputeOperation... forwardComputeOperations) {
        super(null, null, forwardComputeOperations);
        this.source = source;
    }

    @Override
    protected void perform() {
        Matrix result = getResult();
        double max = 0;
        for (int row = 0; row < source.numRows(); row++) {
            for (int col = 0; col < source.numCols(); col++) {
                double sourceVal = source.getValue(row, col);
                if(row == 0 && col == 0) {
                    max = sourceVal;
                } else {
                    max = Math.max(max, sourceVal);
                }
            }
        }
        result.setValue(0, 0, max);
    }
}
