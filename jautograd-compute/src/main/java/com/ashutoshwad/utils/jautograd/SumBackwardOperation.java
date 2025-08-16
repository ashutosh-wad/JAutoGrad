package com.ashutoshwad.utils.jautograd;

class SumBackwardOperation extends BackwardComputeOperation {
    private final Matrix source;

    public SumBackwardOperation(Matrix source, BackwardComputeOperation... backwardComputeOperations) {
        super(null, null, backwardComputeOperations);
        this.source = source;
    }

    @Override
    protected void perform() {
        final double resultGradient = getResult().getGradient();
        for (int row = 0; row < source.numRows(); row++) {
            for (int col = 0; col < source.numCols(); col++) {
                source.accumulateGradient(row, col, resultGradient);
            }
        }
    }
}
