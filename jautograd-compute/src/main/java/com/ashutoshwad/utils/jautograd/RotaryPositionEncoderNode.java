package com.ashutoshwad.utils.jautograd;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class RotaryPositionEncoderNode {
    private static record ThetaCacheKey(int numRows, int numCols) {}
    private static final Map<ThetaCacheKey, MatrixStorage> thetaCache = new ConcurrentHashMap<>();

    private static void computeCaches(ThetaCacheKey key) {
        if (thetaCache.containsKey(key)) {
            return;
        }
        MatrixStorage thetaStore = new MatrixStorage(key.numRows(), key.numCols());
        int dk = key.numCols();
        for (int m = 0; m < key.numRows(); m++) {
            for (int i = 0; i < key.numCols(); i+=2) {
                float theta = m * (float)Math.pow(1.0 / 10000, (double)i / dk);
                thetaStore.set(m, i, (float) Math.sin(theta));
                thetaStore.set(m, i + 1, (float) Math.cos(theta));
            }
        }
        thetaCache.put(key, thetaStore);
    }

    public static class ForwardPass extends ForwardComputeOperation {
        private final Matrix input;
        public ForwardPass(Matrix input, ForwardComputeOperation... forwardComputeOperations) {
            super(null, null, forwardComputeOperations);
            this.input = input;
        }

        @Override
        protected void perform() {
            Matrix result = getResult();

            ThetaCacheKey key = new ThetaCacheKey(input.numRows(), input.numCols());
            computeCaches(key);
            MatrixStorage thetaStore = thetaCache.get(key);

            for (int row = 0; row < key.numRows(); row++) {
                for (int col = 0; col < key.numCols(); col+=2) {
                    float a = input.getValue(row, col);
                    float b = input.getValue(row, col + 1);
                    float sinTheta = thetaStore.get(row, col);
                    float cosTheta = thetaStore.get(row, col + 1);

                    float aSinTheta = a * sinTheta;
                    float aCosTheta = a * cosTheta;
                    float bSinTheta = b * sinTheta;
                    float bCosTheta = b * cosTheta;

                    result.setValue(row, col, aCosTheta + bSinTheta);
                    result.setValue(row, col + 1, bCosTheta - aSinTheta);
                }
            }
        }
    }

    public static class BackwardPass extends BackwardComputeOperation {
        private final Matrix input;

        public BackwardPass(Matrix input, BackwardComputeOperation... backwardComputeOperations) {
            super(null, null, backwardComputeOperations);
            this.input = input;
        }

        @Override
        protected void perform() {
            Matrix result = getResult();

            ThetaCacheKey key = new ThetaCacheKey(input.numRows(), input.numCols());
            computeCaches(key);
            MatrixStorage thetaStore = thetaCache.get(key);

            for (int row = 0; row < key.numRows(); row++) {
                for (int col = 0; col < key.numCols(); col+=2) {
                    float sinTheta = thetaStore.get(row, col);
                    float cosTheta = thetaStore.get(row, col + 1);

                    float gradient1 = result.getGradient(row, col);
                    float gradient2 = result.getGradient(row, col + 1);

                    input.accumulateGradient(row, col, gradient1 * cosTheta - gradient2 * sinTheta);
                    input.accumulateGradient(row, col + 1, gradient1 * sinTheta + gradient2 * cosTheta);
                }
            }
        }
    }
}
