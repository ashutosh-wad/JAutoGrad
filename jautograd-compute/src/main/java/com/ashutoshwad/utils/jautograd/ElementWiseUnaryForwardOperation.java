package com.ashutoshwad.utils.jautograd;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

class ElementWiseUnaryForwardOperation extends ForwardComputeOperation {
    private final FunctionRegistry.UnaryCalcFunction forwardFunction;
    ElementWiseUnaryForwardOperation(Matrix left,
                                      FunctionRegistry.UnaryCalcFunction forwardFunction,
                                      ForwardComputeOperation... forwardComputeOperations) {
        super(left, null, forwardComputeOperations);
        this.forwardFunction = forwardFunction;
    }

    @Override
    protected void perform() {
        ExecutorFactory.Details details = ExecutorFactory.getDetails();
        int numThreads = details.numThreads();
        ExecutorService executorService = details.executorService();

        List<Future<?>>futures = new ArrayList<>(numThreads);
        for (int i = 0; i < numThreads; i++) {
            futures.add(executorService.submit(new ElementWiseUnaryForwardOperation.WorkerJob(getLeft(), getResult(), i, numThreads)));
        }
        ExecutorUtils.awaitFutures(futures);
    }

    private class WorkerJob implements Runnable {
        private final Matrix input;
        private final Matrix result;
        private final int start;
        private final int stride;

        public WorkerJob(Matrix input, Matrix result, int start, int stride) {
            this.input = input;
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
                double inputValue = input.getValue(row, column);
                result.setValue(row, column, forwardFunction.result(inputValue));
            }
        }
    }
}
