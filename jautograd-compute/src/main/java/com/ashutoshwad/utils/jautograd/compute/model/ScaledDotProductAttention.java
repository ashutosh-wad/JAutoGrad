package com.ashutoshwad.utils.jautograd.compute.model;

import com.ashutoshwad.utils.jautograd.compute.ComputeNode;
import com.ashutoshwad.utils.jautograd.compute.Matrix;

import java.util.LinkedList;
import java.util.Queue;

public class ScaledDotProductAttention implements TransformerStage {
    private final int dk;
    private final int dv;
    private final Matrix key;
    private final Matrix key_bias;
    private final Matrix query;
    private final Matrix query_bias;
    private final Matrix value;
    private final Matrix value_bias;

    public ScaledDotProductAttention(int dModel, int dk, int dv) {
        this.dk = dk;
        this.dv = dv;
        this.key = Matrix.createXavierGlorotInitializedMatrix(dModel, dk, true);
        this.key_bias = Matrix.createMatrix(1, dk, ()->0.0, true);
        this.query = Matrix.createXavierGlorotInitializedMatrix(dModel, dk, true);
        this.query_bias = Matrix.createMatrix(1, dk, ()->0.0, true);
        this.value = Matrix.createXavierGlorotInitializedMatrix(dModel, dv, true);
        this.value_bias = Matrix.createMatrix(1, dv, ()->0.0, true);
    }

    @Override
    public Matrix apply(Matrix input) {
        ComputeNode sqrtDk = new ComputeNode(Math.sqrt(dk * 1.0));
        Matrix qPrime = input.matmul(query).add(query_bias);
        Matrix kPrime = input.matmul(key).add(key_bias).transpose();
        Matrix vPrime = input.matmul(value).add(value_bias);

        Matrix qKt = qPrime.matmul(kPrime);
        Matrix scaledQKt = qKt.op(node -> node.div(sqrtDk));

        Matrix maskedQKt = maskedSoftmax(scaledQKt);

        Matrix output = maskedQKt.matmul(vPrime);
        return output;
    }

    private Matrix maskedSoftmax(Matrix qKt) {
        ComputeNode ZERO = new ComputeNode(0.0);
        Matrix maskedQKt = new Matrix(qKt);
        for (int i = 0; i < maskedQKt.rowLength(); i++) {
            //Find softmax of lower left triangle and set the rest to 0
            ComputeNode[]attentionSpan = new ComputeNode[i+1];
            for (int j = 0; j <= i; j++) {
                attentionSpan[j] = maskedQKt.get(i, j);
            }
            attentionSpan = softmax(attentionSpan);
            for (int j = 0; j <= i; j++) {
                maskedQKt.set(i, j, attentionSpan[j]);
            }
            for (int j = i+1; j < maskedQKt.columnLength(); j++) {
                maskedQKt.set(i, j, ZERO);
            }
        }
        return maskedQKt;
    }

    private ComputeNode[] softmax(ComputeNode[] attentionSpan) {
        Queue<ComputeNode> maxQueue = new LinkedList<>();
        for (int i = 0; i < attentionSpan.length; i++) {
            maxQueue.add(attentionSpan[i]);
        }
        while(maxQueue.size() > 1) {
            ComputeNode a = maxQueue.poll();
            ComputeNode b = maxQueue.poll();
            maxQueue.add(a.max(b));
        }
        ComputeNode max = maxQueue.poll();

        for (int i = 0; i < attentionSpan.length; i++) {
            attentionSpan[i] = attentionSpan[i].sub(max);
            attentionSpan[i] = attentionSpan[i].exp();
        }

        Queue<ComputeNode> sumQueue = new LinkedList<>();
        for (int i = 0; i < attentionSpan.length; i++) {
            sumQueue.add(attentionSpan[i]);
        }
        while(sumQueue.size() > 1) {
            ComputeNode a = sumQueue.poll();
            ComputeNode b = sumQueue.poll();
            sumQueue.add(a.add(b));
        }
        ComputeNode sum = sumQueue.poll();

        for (int i = 0; i < attentionSpan.length; i++) {
            attentionSpan[i] = attentionSpan[i].div(sum);
        }

        return attentionSpan;
    }
}
