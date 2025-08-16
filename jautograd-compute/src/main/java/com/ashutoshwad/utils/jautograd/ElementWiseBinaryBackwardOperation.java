package com.ashutoshwad.utils.jautograd;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

public class ElementWiseBinaryBackwardOperation extends BackwardComputeOperation {
    private final FunctionRegistry.BinaryGradientFunction leftBackwardFunction;
    private final FunctionRegistry.BinaryGradientFunction rightBackwardFunction;
    ElementWiseBinaryBackwardOperation(Matrix left,
                                       Matrix right,
                                       FunctionRegistry.BinaryGradientFunction leftBackwardFunction,
                                       FunctionRegistry.BinaryGradientFunction rightBackwardFunction,
                                       BackwardComputeOperation...backwardComputeOperations) {
        super(left, right, backwardComputeOperations);
        this.leftBackwardFunction = leftBackwardFunction;
        this.rightBackwardFunction = rightBackwardFunction;
    }

    @Override
    protected void perform() {
        ExecutorFactory.Details details = ExecutorFactory.getDetails();
        int numThreads = details.numThreads();
        ExecutorService executorService = details.executorService();

        List<Future<?>> futures = new ArrayList<>(details.numThreads());

        final Matrix left = getLeft();
        final Matrix right = getRight();
        final Matrix result = getResult();
        for (int i = 0; i < numThreads; i++) {
            futures.add(executorService.submit(new ElementWiseBinaryBackwardOperation.WorkerJob(left, right, result, i, numThreads)));
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
            final boolean isLeftGradEnabled = left.requiresGradient;
            final boolean isRightGradEnabled = right.requiresGradient;

            for (int i = start; i < totalElements; i+=stride) {
                int row = i / numCols;
                int column = i % numCols;
                double leftValue = left.getValue(row, column);
                double rightValue = right.getValue(row, column);
                double resultValue = result.getValue(row, column);
                double resultGradient = result.getGradient(row, column);
                if (isLeftGradEnabled) {
                    double leftGradient = leftBackwardFunction.result(leftValue, rightValue, resultValue, resultGradient);
                    left.accumulateGradient(row, column, leftGradient);
                }
                if (isRightGradEnabled) {
                    double rightGradient = rightBackwardFunction.result(leftValue, rightValue, resultValue, resultGradient);
                    right.accumulateGradient(row, column, rightGradient);
                }
            }
        }
    }
}
