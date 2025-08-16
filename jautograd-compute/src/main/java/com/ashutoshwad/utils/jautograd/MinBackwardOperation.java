package com.ashutoshwad.utils.jautograd;

import java.util.ArrayList;
import java.util.List;

class MinBackwardOperation extends BackwardComputeOperation {
    private final Matrix source;

    public MinBackwardOperation(Matrix source, BackwardComputeOperation... backwardComputeOperations) {
        super(null, null, backwardComputeOperations);
        this.source = source;
    }

    @Override
    protected void perform() {
        Matrix result = getResult();
        final double min = result.getValue();
        final double resultGradient = result.getGradient();
        int counter = 0;

        List<int[]> minPositions = new ArrayList<>();

        for (int row = 0; row < source.numRows(); row++) {
            for (int col = 0; col < source.numCols(); col++) {
                if(Math.abs(source.getValue(row, col) - min) < Matrix.EPSILON) {
                    counter++;
                    minPositions.add(new int[]{row, col});
                }
            }
        }
        final double revisedGradient = resultGradient / counter;

        for (int[]pos : minPositions) {
            source.accumulateGradient(pos[0], pos[1], revisedGradient);
        }
    }
}
