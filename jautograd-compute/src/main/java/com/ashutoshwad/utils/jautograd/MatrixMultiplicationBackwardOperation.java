package com.ashutoshwad.utils.jautograd;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

public class MatrixMultiplicationBackwardOperation extends BackwardComputeOperation {
    MatrixMultiplicationBackwardOperation(Matrix left,
                                       Matrix right,
                                       BackwardComputeOperation...backwardComputeOperations) {
        super(left, right, backwardComputeOperations);
    }

    @Override
    protected void perform() {
        ExecutorFactory.Details details = ExecutorFactory.getDetails();
        int numThreads = details.numThreads();
        ExecutorService executorService = details.executorService();

        List<Future<?>> futures = new ArrayList<>(numThreads);

        final Matrix left = getLeft();
        final Matrix right = getRight();
        final Matrix result = getResult();
        for (int i = 0; i < numThreads; i++) {
            if(left.getRequiresGradient()) {
                futures.add(executorService.submit(new MatrixMultiplicationBackwardOperation.WorkerJobLeft(left, right, result, i, numThreads)));
            }
            if(right.getRequiresGradient()) {
                futures.add(executorService.submit(new MatrixMultiplicationBackwardOperation.WorkerJobRight(left, right, result, i, numThreads)));
            }
        }

        ExecutorUtils.awaitFutures(futures);
    }

    private class WorkerJobLeft implements Runnable {
        private final Matrix left;
        private final Matrix right;
        private final Matrix result;
        private final int start;
        private final int stride;

        public WorkerJobLeft(Matrix left, Matrix right, Matrix result, int start, int stride) {
            this.left = left;
            this.right = right;
            this.result = result;
            this.start = start;
            this.stride = stride;
        }

        @Override
        public void run() {
            final int numRows = left.numRows();
            final int numCols = left.numCols();
            final int totalElements = numRows * numCols;
            for (int i = start; i < totalElements; i+=stride) {
                final int lRow = i / numCols;
                final int lCol = i % numCols;

                double temp = 0;
                //lRow is selected output row and lCol is selected weight row
                for (int resCol = 0; resCol < result.numCols(); resCol++) {
                    temp += result.getGradient(lRow, resCol) * right.getValue(lCol, resCol);
                }
                left.accumulateGradient(lRow, lCol, temp);
            }
        }
    }

    private class WorkerJobRight implements Runnable {
        private final Matrix left;
        private final Matrix right;
        private final Matrix result;
        private final int start;
        private final int stride;

        public WorkerJobRight(Matrix left, Matrix right, Matrix result, int start, int stride) {
            this.left = left;
            this.right = right;
            this.result = result;
            this.start = start;
            this.stride = stride;
        }

        @Override
        public void run() {
            final int numRows = right.numRows();
            final int numCols = right.numCols();
            final int totalElements = numRows * numCols;
            for (int i = start; i < totalElements; i+=stride) {
                final int rRow = i / numCols;
                final int rCol = i % numCols;

                double temp = 0;
                //rRow is selected input column and rCol is selected output column
                for (int resRow = 0; resRow < result.numRows(); resRow++) {
                    temp += result.getGradient(resRow, rCol) * left.getValue(resRow, rRow);
                }
                right.accumulateGradient(rRow, rCol, temp);
            }
        }
    }
}
