package com.ashutoshwad.utils.jautograd.optimizer;

import com.ashutoshwad.utils.jautograd.ExecutorFactory;
import com.ashutoshwad.utils.jautograd.Matrix;

import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

public class StocasticGradientDescentOptimizer {
    private final List<Matrix> parameters;
    private final double learningRate;

    public StocasticGradientDescentOptimizer(double learningRate) {
        this.parameters = new LinkedList<>();
        this.learningRate = learningRate;
    }

    public void addParameter(Matrix m) {
        parameters.add(m);
    }

    public synchronized void learn() {
        ExecutorFactory.Details details = ExecutorFactory.getDetails();
        ExecutorService executorService = details.executorService();

        List<Future<?>>futures = new LinkedList<>();
        for (Matrix params:parameters) {
            futures.add(executorService.submit(new WorkerJob(params)));
        }
        for (Future<?>future:futures) {
            try {
                future.get();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Worker was interrupted", e);
            } catch (ExecutionException e) {
                throw new RuntimeException("Worker failed to execute", e);
            }
        }
    }

    private class WorkerJob implements Runnable {
        private final Matrix parameters;

        public WorkerJob(Matrix parameters) {
            this.parameters = parameters;
        }

        @Override
        public void run() {
            for (int row = 0; row < parameters.numRows(); row++) {
                for (int col = 0; col < parameters.numCols(); col++) {
                    parameters.setValue(row, col, parameters.getValue(row, col) - (learningRate * parameters.getGradient(row, col)));
                }
            }
        }
    }
}
