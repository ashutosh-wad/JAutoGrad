package com.ashutoshwad.utils.jautograd.optimizer;

import com.ashutoshwad.utils.jautograd.ExecutorFactory;
import com.ashutoshwad.utils.jautograd.ExecutorUtils;
import com.ashutoshwad.utils.jautograd.Matrix;

import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.*;

public class GradientClipper {
    private static final double EPSILON = 0.0000001;
    private final List<Matrix> parameters;

    public GradientClipper(double learningRate) {
        this.parameters = new LinkedList<>();
    }

    public void addParameter(Matrix m) {
        parameters.add(m);
    }

    public void clipGradients(double maxNorm) {
        if (maxNorm <= 0) {
            return;
        }
        Queue<Double>results = new LinkedBlockingQueue<>(parameters.size());
        ExecutorFactory.Details details = ExecutorFactory.getDetails();
        ExecutorService executorService = details.executorService();

        List<Future<?>>futures = new LinkedList<>();
        for (Matrix parameter:parameters) {
            futures.add(executorService.submit(new CalculateCurrentVariance(results, parameter)));
        }
        ExecutorUtils.awaitFutures(futures);

        double totNorm = 0;
        while (results.size()>0) {
            totNorm += results.poll();
        }

        totNorm = Math.sqrt(totNorm);

        if(Math.abs(totNorm)<=EPSILON) {
            return;
        }

        if (totNorm <= maxNorm) {
            return;
        }

        double scale = maxNorm / totNorm;

        futures.clear();
        for (Matrix parameter:parameters) {
            futures.add(executorService.submit(new ClipGradients(parameter, scale)));
        }
        ExecutorUtils.awaitFutures(futures);
    }

    private class ClipGradients implements Runnable {
        private final Matrix parameters;
        private final double scale;

        public ClipGradients(Matrix parameters, double scale) {
            this.parameters = parameters;
            this.scale = scale;
        }

        @Override
        public void run() {
            for (int row = 0; row < parameters.numRows(); row++) {
                for (int col = 0; col < parameters.numCols(); col++) {
                    double gradient = parameters.getGradient(row, col);
                    parameters.setGradient(row, col, gradient * scale);
                }
            }
        }
    }

    private class CalculateCurrentVariance implements Runnable {
        private final Queue<Double> results;
        private final Matrix parameters;

        public CalculateCurrentVariance(Queue<Double> results, Matrix parameters) {
            this.results = results;
            this.parameters = parameters;
        }

        @Override
        public void run() {
            double runningTotal = 0;
            for (int row = 0; row < parameters.numRows(); row++) {
                for (int col = 0; col < parameters.numCols(); col++) {
                    double gradient = parameters.getGradient(row, col);
                    runningTotal += gradient * gradient;
                }
            }
            results.add(runningTotal);
        }
    }
}
