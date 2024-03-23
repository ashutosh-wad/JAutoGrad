package com.ashutoshwad.utils.jautograd;

public interface Value {
	public Value add(Value other);
	public Value sub(Value other);
	public Value div(Value other);
	public Value mul(Value other);
	public Value pow(Value other);
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
	
	
	
	public void forward();
	public void backward();
	public void learn(double rate);
	public void reset();
	public void createDotGraph();
}
