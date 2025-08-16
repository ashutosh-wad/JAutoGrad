package com.ashutoshwad.utils.jautograd;

import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

abstract class ForwardComputeOperation {
    private static final AtomicLong SEQUENCE_FACTORY = new AtomicLong(0);
    private final long id;
    private final Matrix left;
    private final Matrix right;
    private Matrix result;
    private final List<ForwardComputeOperation> forwardComputeOperations;

    public ForwardComputeOperation(Matrix left, Matrix right, ForwardComputeOperation...forwardComputeOperations) {
        this.id = SEQUENCE_FACTORY.getAndIncrement();
        this.left = left;
        this.right = right;
        if (null == forwardComputeOperations || forwardComputeOperations.length == 0) {
            this.forwardComputeOperations = new LinkedList<>();
        } else {
            List<ForwardComputeOperation>temp = new LinkedList<>();
            for (int i = 0; i < forwardComputeOperations.length; i++) {
                if (null != forwardComputeOperations[i]) {
                    temp.add(forwardComputeOperations[i]);
                }
            }
            this.forwardComputeOperations = Collections.unmodifiableList(temp);
        }
    }

    protected final void setResult(Matrix result) {
        if(null==this.result) {
            this.result = result;
        }
        else {
            throw new IllegalArgumentException("Result has been assigned, cannot reassign it");
        }
    }

    protected final Matrix getLeft() {
        return left;
    }

    protected final Matrix getRight() {
        return right;
    }

    protected final Matrix getResult() {
        return result;
    }

    private void collectComputeNodes(Map<Long, ForwardComputeOperation> state) {
        if(state.containsKey(id)) {
            return;
        }
        state.put(id, this);
        forwardComputeOperations.forEach(op -> op.collectComputeNodes(state));
    }

    public final void forward() {
        Map<Long, ForwardComputeOperation> computeNodeMap = new TreeMap<>();
        collectComputeNodes(computeNodeMap);
        computeNodeMap.forEach((k, v) -> v.perform());
    }

    protected final void zeroGradIndividual() {
        if(null == result) {
            return;
        }
        int numRows = result.numRows();
        int numCols = result.numCols();
        for (int row = 0; row < numRows; row++) {
            for (int cols = 0; cols < numCols; cols++) {
                result.setGradient(row, cols, 0);
            }
        }
    }

    public final void zeroGradAndforward() {
        Map<Long, ForwardComputeOperation> computeNodeMap = new TreeMap<>();
        collectComputeNodes(computeNodeMap);
        computeNodeMap.forEach((k, v) -> {
            v.zeroGradIndividual();
            v.perform();
        });
    }

    public final void zeroGrad() {
        Map<Long, ForwardComputeOperation> computeNodeMap = new TreeMap<>();
        collectComputeNodes(computeNodeMap);
        computeNodeMap.forEach((k, v) -> v.zeroGradIndividual());
    }

    protected abstract void perform();
}
