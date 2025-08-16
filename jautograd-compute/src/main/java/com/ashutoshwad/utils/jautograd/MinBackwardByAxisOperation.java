package com.ashutoshwad.utils.jautograd;

import java.util.ArrayList;
import java.util.List;

public class MinBackwardByAxisOperation extends BackwardComputeOperation {
    private final Matrix source;
    private final int axis;
    public MinBackwardByAxisOperation(Matrix source, int axis, BackwardComputeOperation... backwardComputeOperations) {
        super(null, null, backwardComputeOperations);
        this.source = source;
        this.axis = axis;
    }

    @Override
    protected void perform() {
        if(axis == 0) {
            minBackwardAcrossRows();
        } else {
            minBackwardAcrossCols();
        }
    }

    private void minBackwardAcrossRows() {
        Matrix result = getResult();
        for (int col = 0; col < result.numCols(); col++) {
            final double gradient = result.getGradient(0, col);
            final double min = result.getValue(0, col);
            int counter = 0;

            List<int[]>positions = new ArrayList<>();
            for (int row = 0; row < source.numRows(); row++) {
                if (Math.abs(source.getValue(row, col) - min) < Matrix.EPSILON) {
                    counter++;
                    positions.add(new int[]{row, col});
                }
            }

            final double revisedGradient = gradient / counter;
            for (int[]pos:positions) {
                source.accumulateGradient(pos[0], pos[1], revisedGradient);
            }
        }
    }

    private void minBackwardAcrossCols() {
        Matrix result = getResult();
        for (int row = 0; row < result.numRows(); row++) {
            final double gradient = result.getGradient(row, 0);
            final double min = result.getValue(row, 0);
            int counter = 0;

            List<int[]>positions = new ArrayList<>();
            for (int col = 0; col < source.numCols(); col++) {
                if (Math.abs(source.getValue(row, col) - min) < Matrix.EPSILON) {
                    counter++;
                    positions.add(new int[]{row, col});
                }
            }

            final double revisedGradient = gradient / counter;
            for (int[]pos:positions) {
                source.accumulateGradient(pos[0], pos[1], revisedGradient);
            }
        }
    }
}
