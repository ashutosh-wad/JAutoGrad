package com.ashutoshwad.utils.jautograd;

class SumForwardOperation extends ForwardComputeOperation {
    private final Matrix source;
    public SumForwardOperation(Matrix source, ForwardComputeOperation... forwardComputeOperations) {
        super(null, null, forwardComputeOperations);
        this.source = source;
    }

    @Override
    protected void perform() {
        Matrix result = getResult();
        double sum = 0;
        for (int row = 0; row < source.numRows(); row++) {
            for (int col = 0; col < source.numCols(); col++) {
                sum += source.getValue(row, col);
            }
        }
        result.setValue(0, 0, sum);
    }
}
