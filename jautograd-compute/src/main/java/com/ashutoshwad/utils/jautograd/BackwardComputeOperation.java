package com.ashutoshwad.utils.jautograd;

import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

abstract class BackwardComputeOperation {
    private static final AtomicLong SEQUENCE_FACTORY = new AtomicLong(0);
    private final long id;
    private final Matrix left;
    private final Matrix right;
    private Matrix result;
    private final List<BackwardComputeOperation> backwardComputeOperations;

    public BackwardComputeOperation(Matrix left, Matrix right, BackwardComputeOperation...backwardComputeOperations) {
        this.id = SEQUENCE_FACTORY.getAndIncrement();
        this.left = left;
        this.right = right;
        if (null==backwardComputeOperations) {
            this.backwardComputeOperations = new LinkedList<>();
        } else {
            List<BackwardComputeOperation>temp = new LinkedList<>();
            for (int i = 0; i < backwardComputeOperations.length; i++) {
                if (null != backwardComputeOperations[i]) {
                    temp.add(backwardComputeOperations[i]);
                }
            }
            this.backwardComputeOperations = Collections.unmodifiableList(temp);
        }
    }

    public void setResult(Matrix result) {
        this.result = result;
    }

    public Matrix getLeft() {
        return left;
    }

    public Matrix getRight() {
        return right;
    }

    public Matrix getResult() {
        return result;
    }

    private void collectComputeNodes(Map<Long, BackwardComputeOperation> state) {
        if(state.containsKey(id)) {
            return;
        }
        state.put(id, this);
        backwardComputeOperations.forEach(op -> op.collectComputeNodes(state));
    }

    public final void backward() {
        Map<Long, BackwardComputeOperation> computeNodeMap = new TreeMap<>(new Comparator<Long>() {
            @Override
            public int compare(Long o1, Long o2) {
                return o2.compareTo(o1); //Reverse the comparison
            }
        });
        collectComputeNodes(computeNodeMap);
        computeNodeMap.forEach((k, v) -> v.perform());
    }

    protected abstract void perform();
}
