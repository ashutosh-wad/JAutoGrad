package com.ashutoshwad.utils.jautograd.compute;

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

    public JAutogradExecutor(List<ComputeNode[]> computeNodeBatches) {
        this(Runtime.getRuntime().availableProcessors(), computeNodeBatches);
    }

    public JAutogradExecutor(int threadCount, List<ComputeNode[]> computeNodeBatches) {
        this.threadCount = Runtime.getRuntime().availableProcessors();
        this.executorService = Executors.newFixedThreadPool(threadCount);
        this.computeNodeBatches = computeNodeBatches;
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

    public void cleanup() {
        executorService.shutdown();
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
