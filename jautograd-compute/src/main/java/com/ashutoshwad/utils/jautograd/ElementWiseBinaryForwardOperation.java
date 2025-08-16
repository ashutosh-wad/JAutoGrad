package com.ashutoshwad.utils.jautograd;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

class ElementWiseBinaryForwardOperation extends ForwardComputeOperation {
    private final FunctionRegistry.BinaryCalcFunction forwardFunction;
    ElementWiseBinaryForwardOperation(Matrix left,
                                     Matrix right,
                                     FunctionRegistry.BinaryCalcFunction forwardFunction,
                                     ForwardComputeOperation... forwardComputeOperations) {
        super(left, right, forwardComputeOperations);
        this.forwardFunction = forwardFunction;
    }

    @Override
    protected void perform() {
        ExecutorFactory.Details details = ExecutorFactory.getDetails();
        int numThreads = details.numThreads();
        ExecutorService executorService = details.executorService();

        List<Future<?>>futures = new ArrayList<>(numThreads);
        for (int i = 0; i < numThreads; i++) {
            futures.add(executorService.submit(new ElementWiseBinaryForwardOperation.WorkerJob(getLeft(), getRight(), getResult(), i, numThreads)));
        }
        ExecutorUtils.awaitFutures(futures);
    }

    private class WorkerJob implements Runnable {
        private final Matrix left;
        private final Matrix right;
        private final Matrix result;
        private final int start;
        private final int stride;

        public WorkerJob(Matrix left, Matrix right, Matrix result, int start, int stride) {
            this.left = left;
            this.right = right;
            this.result = result;
            this.start = start;
            this.stride = stride;
        }

        @Override
        public void run() {
            int numRows = result.numRows();
            final int numCols = result.numCols();
            final int totalElements = numRows * numCols;

            for (int i = start; i < totalElements; i+=stride) {
                int row = i / numCols;
                int column = i % numCols;
                double leftValue = left.getValue(row, column);
                double rightValue = right.getValue(row, column);
                result.setValue(row, column, forwardFunction.result(leftValue, rightValue));
            }
        }
    }
}
