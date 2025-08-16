package com.ashutoshwad.utils.jautograd.optimizer;

import com.ashutoshwad.utils.jautograd.ExecutorFactory;
import com.ashutoshwad.utils.jautograd.Matrix;

import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

public class AdamOptimizer {
    private record GradientData(Matrix parameters, double[][]momentum, double[][]variance){}

    private static final double EPSILON = 0.0000001;

    private final double MOMENTUM_BETA;
    private final double VARIANCE_BETA;
    private final double learningRate;
    private final List<GradientData> gradientDataList;

    private int step;

    public AdamOptimizer(double learningRate) {
        this(learningRate, 0.9, 0.999);
    }

    public AdamOptimizer(double learningRate, double momentumBeta, double varianceBeta) {
        this.gradientDataList = new LinkedList<>();
        this.learningRate = learningRate;
        this.step = 0;
        this.MOMENTUM_BETA = momentumBeta;
        this.VARIANCE_BETA = varianceBeta;
    }

    public void addParameter(Matrix m) {
        GradientData data = new GradientData(m, new double[m.numRows()][m.numCols()], new double[m.numRows()][m.numCols()]);
        gradientDataList.add(data);
    }

    public synchronized void learn() {
        step++;
        ExecutorFactory.Details details = ExecutorFactory.getDetails();
        ExecutorService executorService = details.executorService();

        List<Future<?>>futures = new LinkedList<>();
        for (GradientData data:gradientDataList) {
            futures.add(executorService.submit(new WorkerJob(data, step)));
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
        private final GradientData data;
        private final int step;

        public WorkerJob(GradientData data, int step) {
            this.data = data;
            this.step = step;
        }

        @Override
        public void run() {
            Matrix parameters = data.parameters;
            for (int row = 0; row < parameters.numRows(); row++) {
                for (int col = 0; col < parameters.numCols(); col++) {
                    final double gradient = data.parameters.getGradient(row, col);
                    double momentum = data.momentum[row][col];
                    double variance = data.variance[row][col];
                    momentum = MOMENTUM_BETA * momentum + (1 - MOMENTUM_BETA) * gradient;
                    variance = VARIANCE_BETA * variance + (1 - VARIANCE_BETA) * (gradient * gradient);
                    data.momentum[row][col] = momentum;
                    data.variance[row][col] = variance;
                    double mPrime = momentum / (1 - Math.pow(MOMENTUM_BETA, step));
                    double vPrime = variance / (1 - Math.pow(VARIANCE_BETA, step));

                    double updatedValue = parameters.getValue(row, col) - learningRate * mPrime / (Math.sqrt(vPrime) + EPSILON);
                    parameters.setValue(row, col, updatedValue);
                }
            }
        }
    }
}
