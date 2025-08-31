package com.ashutoshwad.utils.jautograd.transformer;

import com.ashutoshwad.utils.jautograd.Matrix;

public class ToyTransformer {
    private Matrix input;
    private Matrix featureMatrix;
    private final int tokenSize;
    private final int numAttentionHeads;
    private final int featureSize;
    private final int keySize;
    private final int valueSize;

    public ToyTransformer() {
        this.tokenSize = 88;
        this.numAttentionHeads = 2;
        this.featureSize = 64;
        this.keySize = 64;
        this.valueSize = 64;
    }

    public Matrix create(int contextSize) {
        input = Matrix.create(contextSize, tokenSize, ()->0.0,false);
        featureMatrix = Matrix.createXavierGlorotInitializedMatrix(tokenSize, featureSize, true);

        Matrix layerInput = input.matmul(featureMatrix);

        TransformerStage stage1 = new TransformerStage(tokenSize, numAttentionHeads, featureSize, keySize, valueSize);
        layerInput = stage1.apply(layerInput);

        /*
        TransformerStage stage2 = new TransformerStage(tokenSize, numAttentionHeads, featureSize, keySize, valueSize);
        layerInput = stage2.apply(layerInput);

        TransformerStage stage3 = new TransformerStage(tokenSize, numAttentionHeads, featureSize, keySize, valueSize);
        layerInput = stage3.apply(layerInput);

        TransformerStage stage4 = new TransformerStage(tokenSize, numAttentionHeads, featureSize, keySize, valueSize);
        layerInput = stage4.apply(layerInput);

        TransformerStage stage5 = new TransformerStage(tokenSize, numAttentionHeads, featureSize, keySize, valueSize);
        layerInput = stage5.apply(layerInput);

        TransformerStage stage6 = new TransformerStage(tokenSize, numAttentionHeads, featureSize, keySize, valueSize);
        layerInput = stage6.apply(layerInput);*/

        /*More layers to be added here*/

        TransformerOutputDistribution dist = new TransformerOutputDistribution(featureSize, tokenSize);
        Matrix layerOutput = dist.apply(layerInput);

        return layerOutput;
    }

    private class TransformerStage {
        private final int tokenSize;
        private final int numAttentionHeads;
        private final int featureSize;
        private final int keySize;
        private final int valueSize;

        private MultiHeadedAttention multiHeadedAttention;
        private LayerNormalizer layerNormalizer1;
        private FeedForwardLayer feedForwardLayer;
        private LayerNormalizer layerNormalizer2;

        public TransformerStage(int tokenSize, int numAttentionHeads, int featureSize, int keySize, int valueSize) {
            this.tokenSize = tokenSize;
            this.numAttentionHeads = numAttentionHeads;
            this.featureSize = featureSize;
            this.keySize = keySize;
            this.valueSize = valueSize;

            this.multiHeadedAttention = new MultiHeadedAttention(numAttentionHeads, featureSize, keySize, valueSize, false, 0);
            this.layerNormalizer1 = new LayerNormalizer(featureSize);
            this.feedForwardLayer = new FeedForwardLayer(featureSize);
            this.layerNormalizer2 = new LayerNormalizer(featureSize);
        }

        public Matrix apply(Matrix input) {
            Matrix normalizedInput = layerNormalizer1.apply(input);
            Matrix attention = multiHeadedAttention.apply(normalizedInput).add(input);
            Matrix renorm = layerNormalizer2.apply(attention);
            Matrix payAttention = feedForwardLayer.apply(renorm, false, 0).add(attention);
            return payAttention;
        }
    }
}
