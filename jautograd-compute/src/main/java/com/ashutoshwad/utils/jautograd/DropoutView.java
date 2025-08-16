package com.ashutoshwad.utils.jautograd;

import java.util.BitSet;
import java.util.Random;
import java.util.function.Supplier;

public class DropoutView extends Matrix {
    private final Matrix source;
    private final BitSet flags;
    private final Supplier<Double> probabilitySupplier;
    private double scale;

    public DropoutView(
            Matrix source,
            Supplier<Double> probabilitySupplier) {
        super(source.requiresGradient,
                new DropoutForwardComputeOperation(source.forwardComputeOperation),
                source.backwardComputeOperation);
        DropoutForwardComputeOperation temp = (DropoutForwardComputeOperation)this.forwardComputeOperation;
        temp.setSource(this);
        this.source = source;
        this.probabilitySupplier = probabilitySupplier;
        this.flags = new BitSet(source.numRows() * source.numCols());
        this.scale = 0;
        temp.forward();
    }

    private static final class DropoutForwardComputeOperation extends ForwardComputeOperation {
        private DropoutView source;
        public DropoutForwardComputeOperation(ForwardComputeOperation... forwardComputeOperations) {
            super(null, null, forwardComputeOperations);
        }

        public void setSource(DropoutView source) {
            if (null == source) {
                return;
            }
            this.source = source;
        }

        @Override
        protected void perform() {
            source.perform();
        }
    }

    protected void perform() {
        flags.clear();
        Random r = new Random();
        double probability = probabilitySupplier.get();
        probability = Math.min(1, probability);
        probability = Math.max(0, probability);
        scale = 1 / (1 - probability);
        int index = 0;
        for (int row = 0; row < source.numRows(); row++) {
            for (int col = 0; col < source.numCols(); col++) {
                double toss = r.nextDouble();
                if (toss < probability) {
                    flags.set(index);
                }
                index++;
            }
        }
    }

    private boolean isDropout(int row, int column) {
        return flags.get(row * source.numCols() + column);
    }

    @Override
    public double getValue(int row, int column) {
        return isDropout(row, column) ? 0 : source.getValue(row, column) * scale;
    }

    @Override
    public synchronized void setValue(int row, int column, double value) {
        if(!isDropout(row, column)) {
            source.setValue(row, column, value);
        }
    }

    @Override
    public double getGradient(int row, int column) {
        if (isDropout(row, column)) {
            return 0;
        } else {
            return source.getGradient(row, column);
        }
    }

    @Override
    public synchronized void setGradient(int row, int column, double value) {
        if (isDropout(row, column)) {
            return;
        }
        source.setGradient(row, column, value);
    }

    @Override
    public synchronized void accumulateGradient(int row, int column, double value) {
        if (isDropout(row, column)) {
            return;
        }
        source.accumulateGradient(row, column, value);
    }

    @Override
    public int numRows() {
        return source.numRows();
    }

    @Override
    public int numCols() {
        return source.numCols();
    }

    @Override
    public boolean getRequiresGradient() {
        return source.getRequiresGradient();
    }
}
