package com.ashutoshwad.utils.jautograd;

public interface Value {
	public Value add(Value other);
	public Value sub(Value other);
	public Value div(Value other);
	public Value mul(Value other);
	public Value pow(Value other);
	public Value sqrt();
	public Value relu();
	public Value relu(double negSlope);
	public Value sinh();
	public Value cosh();
	public Value tanh();
	public Value sin();
	public Value cos();
	public Value tan();
	public Value sigmoid();
	public Value exponential();
	public Value ln();

	public double getValue();
	public void setValue(double value);
	public double getGradient();

	public void forward();
	public void backward();
	public void learn(double rate);
	public void reset();
	public String createDotGraph();

	public static Value of(double value) {
		return new JAutogradValue(value);
	}
	public static Value learnable(double value) {
		return new JAutogradValue(value, true);
	}
}
