package com.ashutoshwad.toynet;

import java.util.Objects;

import com.ashutoshwad.utils.jautograd.Value;

public class ToyNetwork {
	private int[] layerdef;
	public ToyNetwork(int...layerdef) {
		Objects.requireNonNull(layerdef);
		if(layerdef.length<2) {
			throw new RuntimeException("A neural net definition requires at least one layer.");
		}
		this.layerdef = layerdef;
	}
	public Value[] connect(Value[]input) {
		Value[]output = null;
		if(input.length != layerdef[0]) {
			throw new RuntimeException("Input must be of length: " + layerdef[0]);
		}
		for (int layer_index = 0; layer_index < layerdef.length - 1; layer_index++) {
			ToyNetLayer layer = new ToyNetLayer(layerdef[layer_index], layerdef[layer_index + 1]);
			if(0 == layer_index) {
				output = layer.connect(input);
			} else {
				output = layer.connect(output);
			}
		}
		return output;
	}
}
