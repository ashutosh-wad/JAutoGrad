package com.ashutoshwad.toynet;

import com.ashutoshwad.utils.jautograd.Value;

public class ToyNetLayer {
	private ToyNetNeuron[] neurons;
	public ToyNetLayer(int numInput, int numOutput) {
		this.neurons = new ToyNetNeuron[numOutput];
		for (int i = 0; i < neurons.length; i++) {
			neurons[i] = new ToyNetNeuron(numInput);
		}
	}
	public Value[] connect(Value[]input) {
		Value[]output = new Value[neurons.length];
		for (int i = 0; i < output.length; i++) {
			output[i] = neurons[i].connect(input);
		}
		return output;
	}
}
