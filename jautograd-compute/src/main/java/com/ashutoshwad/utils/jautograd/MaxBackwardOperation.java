package com.ashutoshwad.utils.jautograd;

import java.util.ArrayList;
import java.util.List;

class MaxBackwardOperation extends BackwardComputeOperation {
    private final Matrix source;

    public MaxBackwardOperation(Matrix source, BackwardComputeOperation... backwardComputeOperations) {
        super(null, null, backwardComputeOperations);
        this.source = source;
    }

    @Override
    protected void perform() {
        Matrix result = getResult();
        final double max = result.getValue();
        final double resultGradient = result.getGradient();
        int counter = 0;

        List<int[]> maxPositions = new ArrayList<>();

        for (int row = 0; row < source.numRows(); row++) {
            for (int col = 0; col < source.numCols(); col++) {
                if(Math.abs(source.getValue(row, col) - max) < Matrix.EPSILON) {
                    counter++;
                    maxPositions.add(new int[]{row, col});
                }
            }
        }
        final double revisedGradient = resultGradient / counter;

        for (int[]pos : maxPositions) {
            source.accumulateGradient(pos[0], pos[1], revisedGradient);
        }
    }
}
