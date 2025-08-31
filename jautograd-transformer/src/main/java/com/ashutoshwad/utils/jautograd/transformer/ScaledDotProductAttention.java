package com.ashutoshwad.utils.jautograd.transformer;

import com.ashutoshwad.utils.jautograd.Matrix;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class ScaledDotProductAttention {
    private record Position(int numRows, int numCols) {}
    private static final Map<Integer, Matrix> maskMap = new ConcurrentHashMap<>();
    private static final Map<Position, RotaryPositionEncoder> ropeMap = new ConcurrentHashMap<>();
    private final Matrix key;
    private final Matrix keyBias;
    private final Matrix query;
    private final Matrix queryBias;
    private final Matrix value;
    private final Matrix valueBias;
    private final Matrix scale;
    private final boolean dropout;
    private final double dropoutRate;

    public ScaledDotProductAttention(int featureSize, int keyDimensionSize, int valueDimensionSize, boolean dropout, double dropoutRate) {
        this.key = Matrix.createXavierGlorotInitializedMatrix(featureSize, keyDimensionSize, true);
        this.keyBias = Matrix.create(1, keyDimensionSize, true);
        this.query = Matrix.createXavierGlorotInitializedMatrix(featureSize, keyDimensionSize, true);
        this.queryBias = Matrix.create(1, keyDimensionSize, true);
        this.value = Matrix.createXavierGlorotInitializedMatrix(featureSize, valueDimensionSize, true);
        this.valueBias = Matrix.create(1, valueDimensionSize, true);
        this.scale = Matrix.create(Math.sqrt(keyDimensionSize));
        this.dropout = dropout;
        this.dropoutRate = dropoutRate;
    }

    public Matrix apply(Matrix input) {
        Matrix queryPrime = input.matmul(query).add(queryBias);
        Matrix keyPrime = input.matmul(key).add(keyBias);
        RotaryPositionEncoder RoPE = ropeMap.computeIfAbsent(new Position(queryPrime.numRows(), queryPrime.numCols()), pos->new RotaryPositionEncoder(pos.numRows(), pos.numCols()));
        queryPrime = RoPE.rotate(queryPrime);
        keyPrime = RoPE.rotate(keyPrime);
        Matrix valuePrime = input.matmul(value).add(valueBias);
        Matrix attention = queryPrime.matmul(keyPrime.transpose()).div(scale);
        Matrix attentionSoftmax = maskedStableSoftmax(attention);
        if(dropout) {
            attentionSoftmax = attentionSoftmax.dropout(dropoutRate);
        }
        return attentionSoftmax.matmul(valuePrime);
    }

    public Matrix maskedStableSoftmax(Matrix attention) {
        Matrix maxVal = attention.max(1);
        Matrix shifted = attention.sub(maxVal);
        Matrix shiftedExp = shifted.exp();
        Matrix maskedAttention = shiftedExp.mul(maskMap.computeIfAbsent(attention.numRows(), ScaledDotProductAttention::createMask));
        Matrix sum = maskedAttention.sum(1);
        return maskedAttention.div(sum.add(Matrix.create(Matrix.EPSILON)));
    }

    private static Matrix createMask(int featureSize) {
        Matrix mask = Matrix.create(featureSize, featureSize, false);
        for (int row = 0; row < mask.numRows(); row++) {
            for (int col = 0; col < mask.numCols(); col++) {
                mask.setValue(row, col, col>row ? 0:1);
            }
        }
        return mask;
    }
}
