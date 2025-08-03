package com.ashutoshwad.utils.jautograd.compute;

import com.ashutoshwad.utils.jautograd.compute.optimizer.LearningOptimizer;
import com.ashutoshwad.utils.jautograd.compute.optimizer.LearningOptimizerType;
import com.ashutoshwad.utils.jautograd.compute.optimizer.StochasticGradientDescentOptimizer;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Consumer;

public class JAutogradExecutor {
    private final int threadCount;
    private final ExecutorService executorService;
    private final List<ComputeNode[]> computeNodeBatches;
    private final List<LearningOptimizer> optimizers;

    public JAutogradExecutor(List<ComputeNode[]> computeNodeBatches) {
        this(Runtime.getRuntime().availableProcessors(), computeNodeBatches);
    }

    public JAutogradExecutor(int threadCount, List<ComputeNode[]> computeNodeBatches) {
        this.threadCount = threadCount;
        this.executorService = Executors.newFixedThreadPool(threadCount);
        this.computeNodeBatches = computeNodeBatches;
        this.optimizers = new LinkedList<>();
    }

    public void zeroGrad() {
        forwardPass(computeNode -> computeNode.setGradient(0));
    }
    public void forward() {
        forwardPass(ComputeNode::calc);
    }
    public void zeroGradAndForward() {
        forwardPass(computeNode -> {
            computeNode.setGradient(0);
            computeNode.calc();
        });
    }

    private void forwardPass(Consumer<ComputeNode> action) {
        for (int i = 0; i < computeNodeBatches.size(); i++) {
            ComputeNode[]batch = computeNodeBatches.get(i);
            List<Future<?>>futures = new LinkedList<>();
            for (int j = 0; j < threadCount; j++) {
                futures.add(executorService.submit(new ComputeNodeExecutor(batch, j, threadCount, action)));
            }
            for(Future<?>future:futures) {
                try {
                    future.get();
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        }
    }

    public void backward() {
        for (int i = computeNodeBatches.size()-1; i >= 0; i--) {
            ComputeNode[]batch = computeNodeBatches.get(i);
            List<Future<?>>futures = new LinkedList<>();
            for (int j = 0; j < threadCount; j++) {
                futures.add(executorService.submit(new ComputeNodeExecutor(batch, j, threadCount, ComputeNode::computeGradient)));
            }
            for(Future<?>future:futures) {
                try {
                    future.get();
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        }
    }

    public void op(Consumer<ComputeNode>operation) {
        forwardPass(operation);
    }

    public void clipGradients(final double maxNorm) {
        double totalNorm = 0.0;
        for (ComputeNode[] batch : computeNodeBatches) {
            for (ComputeNode node : batch) {
                double grad = node.getGradient();
                totalNorm += grad * grad;
            }
        }
        totalNorm = Math.sqrt(totalNorm);

        if (totalNorm > maxNorm) {
            double scale = maxNorm / totalNorm;
            for (ComputeNode[] batch : computeNodeBatches) {
                for (ComputeNode node : batch) {
                    node.setGradient(node.getGradient() * scale);
                }
            }
        }
    }

    public void initializeOptimizer(LearningOptimizerType type) {
        optimizers.clear();
        for (ComputeNode[]computeNode:computeNodeBatches) {
            for (ComputeNode node:computeNode) {
                switch (type) {
                    case SGD:
                        optimizers.add(new StochasticGradientDescentOptimizer(node));
                        break;
                    case ADAM: throw new UnsupportedOperationException("ADAM is not yet supported");
                    default:
                        throw new UnsupportedOperationException(type + " is not yet supported");
                }
            }
        }
    }

    public void learn(double learningRate) {
        if(optimizers.size() == 0) {
            throw new RuntimeException("You need to call `initializeOptimizer` first to set the optimizer type");
        }
        final int batchSize = 10000 * threadCount;
        List<LearningOptimizer>batch = new ArrayList<>(batchSize);
        for (LearningOptimizer opt:optimizers) {
            batch.add(opt);
            if(batch.size() == batchSize) {
                executeOptimizersInParallel(batch, learningRate);
                batch.clear();
            }
        }
        if(!batch.isEmpty()) {
            executeOptimizersInParallel(batch, learningRate);
            batch.clear();
        }
    }

    private void executeOptimizersInParallel(List<LearningOptimizer>batch, double learningRate) {
        List<Future<?>>futures = new ArrayList<>();
        for (int i = 0; i < threadCount; i++) {
            futures.add(executorService.submit(new OptimizerExecutor(batch, i, threadCount, learningRate)));
        }
        for (int i = 0; i < threadCount; i++) {
            try {
                futures.get(i).get();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }

    public void cleanup() {
        executorService.shutdown();
    }

    private static final class OptimizerExecutor implements Runnable {
        private final List<LearningOptimizer> optimizers;
        private final int start;
        private final int stride;
        private final double learningRate;

        public OptimizerExecutor(List<LearningOptimizer> optimizers, int start, int stride, double learningRate) {
            this.optimizers = optimizers;
            this.start = start;
            this.stride = stride;
            this.learningRate = learningRate;
        }

        @Override
        public void run() {
            for (int i = start; i < optimizers.size(); i+=stride) {
                optimizers.get(i).learn(learningRate);
            }
        }
    }
    private static final class ComputeNodeExecutor implements Runnable {
        private final ComputeNode[] computeNodes;
        private final int start;
        private final int stride;
        private final Consumer<ComputeNode> action;

        public ComputeNodeExecutor(ComputeNode[] computeNodes, int start, int stride, Consumer<ComputeNode> action) {
            this.computeNodes = computeNodes;
            this.start = start;
            this.stride = stride;
            this.action = action;
        }

        @Override
        public void run() {
            for (int i = start; i < computeNodes.length; i+=stride) {
                action.accept(computeNodes[i]);
            }
        }
    }
}
