package com.ashutoshwad.utils.jautograd.toynet;

import java.util.LinkedList;
import java.util.Queue;
import java.util.Random;

import com.ashutoshwad.utils.jautograd.Value;

public class ToyNetNeuron {
	private Value[] weights;
	private Value bias;
	public ToyNetNeuron(int numInput) {
		this.weights = new Value[numInput];
		for (int i = 0; i < weights.length; i++) {
			this.weights[i] = Value.learnable(generateRandomWeight());
		}
		this.bias = Value.learnable(generateRandomWeight());
	}
	private double generateRandomWeight() {
		Random r = new Random();
		double next = (r.nextDouble() * 2.0) - 1;
		return next;
	}
	public Value connect(Value[]input) {
		if(weights.length!=input.length) {
			throw new RuntimeException("Input array expected to have length: " + weights.length);
		}
		Queue<Value>resultQueue = new LinkedList<Value>();
		for (int i = 0; i < input.length; i++) {
			resultQueue.add(weights[i].mul(input[i]));
		}
		resultQueue.add(bias);
		Value sum = sum(resultQueue);
		sum = sum.tanh();
		return sum;
	}
	private Value sum(Queue<Value> resultQueue) {
		while(resultQueue.size()>1) {
			Value v1 = resultQueue.poll();
			Value v2 = resultQueue.poll();
			resultQueue.add(v1.add(v2));
		}
		return resultQueue.poll();
	}
}
