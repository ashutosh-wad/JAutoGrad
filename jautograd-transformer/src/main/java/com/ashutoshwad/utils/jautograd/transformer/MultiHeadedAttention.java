package com.ashutoshwad.utils.jautograd.transformer;

import com.ashutoshwad.utils.jautograd.Matrix;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class MultiHeadedAttention {
    private List<ScaledDotProductAttention> heads;
    private Matrix valueDown;
    private Matrix valueDownBias;

    public MultiHeadedAttention(int numHeads, int featureSize, int keyDimensionSize, int valueDimensionSize, boolean dropout, double dropoutRate) {
        heads = new ArrayList<>(numHeads);
        for (int i = 0; i < numHeads; i++) {
            heads.add(new ScaledDotProductAttention(featureSize, keyDimensionSize, valueDimensionSize, dropout, dropoutRate));
        }
        valueDown = Matrix.createXavierGlorotInitializedMatrix(numHeads * valueDimensionSize, featureSize, true);
        valueDownBias = Matrix.createXavierGlorotInitializedMatrix(1, featureSize, true);
    }

    public Matrix apply(Matrix input) {
        Queue<Matrix> valueOutQueue = new LinkedList<>();
        for(ScaledDotProductAttention head : heads) {
            valueOutQueue.add(head.apply(input));
        }
        Matrix valueConcat = getConcatenatedValueMatrix(valueOutQueue);
        return valueConcat.matmul(valueDown).add(valueDownBias);
    }

    private Matrix getConcatenatedValueMatrix(Queue<Matrix> valueOutQueue) {
        Matrix first = valueOutQueue.poll();
        if(valueOutQueue.size() == 0) {
            return first;
        }
        Matrix[]valueOutSuffix = new Matrix[valueOutQueue.size()];
        for (int i = 0; i < valueOutSuffix.length; i++) {
            valueOutSuffix[i] = valueOutQueue.poll();
        }
        return first.concatHorizontal(valueOutSuffix);
    }
}
