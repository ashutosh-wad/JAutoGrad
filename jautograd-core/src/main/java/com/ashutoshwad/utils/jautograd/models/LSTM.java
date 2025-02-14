package com.ashutoshwad.utils.jautograd.models;

import com.ashutoshwad.utils.jautograd.Value;

public class LSTM {
    private Value longTermMemory;
    private Value shortTermMemory;
    private Value[]weights;
    private Value[]biases;

    public LSTM(Value longTermMemory, Value shortTermMemory, Value[] weights, Value[] biases) {
        this.longTermMemory = validateValue(longTermMemory, "LSTM longTermMemory");
        this.shortTermMemory = validateValue(shortTermMemory, "LSTM shortTermMemory");
        this.weights = validateWeights(weights, 8, "LSTM weights");
        this.biases = validateWeights(biases, 4, "LSTM weights");
    }

    private Value validateValue(Value value, String name) {
        if(null == value) {
            throw new NullPointerException(name + " cannot be null");
        }
        return value;
    }

    private Value[] validateWeights(Value[]values, int requiredLength, String name) {
        if(null == values) {
            throw new NullPointerException(name + " cannot be null");
        }
        if(requiredLength != values.length) {
            throw new IllegalArgumentException(name + "must have a length of "+requiredLength+ " but found " + values.length);
        }
        for (int i = 0; i < values.length; i++) {
            validateValue(values[i], name + " at position " + i);
        }
        return values;
    }

    public Value forward(Value[]input) {
        Value output = null;
        for (int i = 0; i < input.length; i++) {
            output = buildLstmSequence(input[i]);
        }
        return output;
    }

    private Value buildLstmSequence(Value input) {
        //Forget gate
        Value forgetGateSigmoid = input.mul(weights[0]).add(shortTermMemory.mul(weights[1])).add(biases[0]);
        forgetGateSigmoid = forgetGateSigmoid.sigmoid();
        longTermMemory = longTermMemory.mul(forgetGateSigmoid);

        // Input gate
        Value rememberGateSigmoid = input.mul(weights[2]).add(shortTermMemory.mul(weights[3])).add(biases[1]);
        rememberGateSigmoid = rememberGateSigmoid.sigmoid();
        Value rememberGateTanh =  input.mul(weights[4]).add(shortTermMemory.mul(weights[5])).add(biases[2]);
        rememberGateTanh = rememberGateTanh.tanh();
        Value rememberGate = rememberGateSigmoid.mul(rememberGateTanh);
        longTermMemory = longTermMemory.add(rememberGate);

        //Output gate
        Value shortTermMemorySigmoid = input.mul(weights[6]).add(shortTermMemory.mul(weights[7])).add(biases[3]);
        shortTermMemorySigmoid = shortTermMemorySigmoid.sigmoid();
        shortTermMemory = longTermMemory.tanh().mul(shortTermMemorySigmoid);
        return shortTermMemory;
    }
}