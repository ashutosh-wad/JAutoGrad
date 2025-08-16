package com.ashutoshwad.utils.jautograd;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

public class ElementWiseUnaryBackwardOperation extends BackwardComputeOperation {
    private final FunctionRegistry.UnaryGradientFunction backwardFunction;
    ElementWiseUnaryBackwardOperation(Matrix left,
                                     FunctionRegistry.UnaryGradientFunction backwardFunction,
                                     BackwardComputeOperation...backwardComputeOperations) {
        super(left, null, backwardComputeOperations);
        this.backwardFunction = backwardFunction;
    }

    @Override
    protected void perform() {
        ExecutorFactory.Details details = ExecutorFactory.getDetails();
        int numThreads = details.numThreads();
        ExecutorService executorService = details.executorService();

        List<Future<?>> futures = new ArrayList<>(numThreads);
        for (int i = 0; i < numThreads; i++) {
            futures.add(executorService.submit(new ElementWiseUnaryBackwardOperation.WorkerJob(getLeft(), getResult(), i, numThreads)));
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
                double resultValue = result.getValue(row, column);
                double resultGradient = result.getGradient(row, column);
                double inputGradient = backwardFunction.result(inputValue, resultValue, resultGradient);
                input.accumulateGradient(row, column, inputGradient);
            }
        }
    }
}
