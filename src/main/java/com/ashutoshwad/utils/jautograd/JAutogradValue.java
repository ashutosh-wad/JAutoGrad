package com.ashutoshwad.utils.jautograd;

import java.io.ByteArrayOutputStream;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.UUID;

import com.ashutoshwad.utils.jautograd.exception.JAutogradException;

public class JAutogradValue implements Value {
	public static enum ValueType {
		VALUE, LEARNABLE, ADD, SUB, DIV, MUL, POW, SIN, COS, TAN, SINH, COSH, TANH, RELU, EXPONENTIAL
	}

	private final String id;
	private final JAutogradValue left;
	private final JAutogradValue right;
	private final ValueType type;
	private final double negSlope;
	private double value;
	private double grad;
	private double gradCount;
	private JAutogradValue[] values;

	public JAutogradValue() {
		this(0.0);
	}

	public JAutogradValue(double value) {
		this(value, false);
	}

	public JAutogradValue(double value, boolean learnable) {
		this(null, null, (learnable) ? ValueType.LEARNABLE : ValueType.VALUE);
		this.value = value;
	}

	private JAutogradValue(Value left, Value right, ValueType type) {
		this(left, right, type, 0);
	}

	private JAutogradValue(Value left, Value right, ValueType type, double negSlope) {
		validateValue(left);
		validateValue(right);
		this.id = "jautograd-" + UUID.randomUUID().toString();
		this.left = (JAutogradValue) left;
		this.right = (JAutogradValue) right;
		this.type = type;
		this.grad = 0;
		this.gradCount = 0;
		this.negSlope = negSlope;
		this.values = null;
		// The following calc value call is important as it correctly initializes the
		// value for this node.
		calcValue();
	}

	private void validateValue(Value val) {
		if (null == val) {
			return;
		}
		if (val instanceof JAutogradValue) {
			return;
		}
		throw new JAutogradException("Unsupported value type: " + val.getClass().getName());
	}

	public String getId() {
		return id;
	}

	@Override
	public double getValue() {
		return value;
	}

	public void setValue(double value) {
		this.value = value;
	}

	@Override
	public double getGradient() {
		return grad;
	}

	public void setGradient(double grad) {
		this.grad = grad;
	}

	@Override
	public Value add(Value other) {
		return new JAutogradValue(this, other, ValueType.ADD);
	}

	@Override
	public Value sub(Value other) {
		return new JAutogradValue(this, other, ValueType.SUB);
	}

	@Override
	public Value div(Value other) {
		return new JAutogradValue(this, other, ValueType.DIV);
	}

	@Override
	public Value mul(Value other) {
		return new JAutogradValue(this, other, ValueType.MUL);
	}

	@Override
	public Value pow(Value other) {
		return new JAutogradValue(this, other, ValueType.POW);
	}

	@Override
	public Value relu() {
		return relu(0);
	}

	@Override
	public Value relu(double negSlope) {
		JAutogradValue temp = new JAutogradValue(this, null, ValueType.RELU, negSlope);
		return temp;
	}

	@Override
	public Value sinh() {
		return new JAutogradValue(this, null, ValueType.SINH);
	}

	@Override
	public Value cosh() {
		return new JAutogradValue(this, null, ValueType.COSH);
	}

	@Override
	public Value tanh() {
		return new JAutogradValue(this, null, ValueType.TANH);
	}

	@Override
	public Value sin() {
		return new JAutogradValue(this, null, ValueType.SIN);
	}

	@Override
	public Value cos() {
		return new JAutogradValue(this, null, ValueType.COS);
	}

	@Override
	public Value tan() {
		return new JAutogradValue(this, null, ValueType.TAN);
	}

	@Override
	public Value sigmoid() {
		Value x = this;
		Value ONE = new JAutogradValue(1);
		return ONE.div(ONE.add(ONE.div(x.exponential())));
	}

	@Override
	public Value exponential() {
		return new JAutogradValue(this, null, ValueType.EXPONENTIAL);
	}

	private void calcValue() {
		switch (type) {
			case VALUE:
			case LEARNABLE:
				break;
			case ADD:
				value = left.getValue() + right.getValue();
				break;
			case SUB:
				value = left.getValue() - right.getValue();
				break;
			case DIV:
				value = left.getValue() / right.getValue();
				break;
			case MUL:
				value = left.getValue() * right.getValue();
				break;
			case POW:
				value = Math.pow(left.getValue(), right.getValue());
				break;
			case SIN:
				value = Math.sin(left.getValue());
				break;
			case COS:
				value = Math.cos(left.getValue());
				break;
			case TAN:
				value = Math.tan(left.getValue());
				break;
			case SINH:
				value = Math.sinh(left.getValue());
				break;
			case COSH:
				value = Math.cosh(left.getValue());
				break;
			case TANH:
				value = Math.tanh(left.getValue());
				break;
			case RELU:
				value = (left.getValue() >= 0) ? left.getValue() : negSlope * left.getValue();
				break;
			case EXPONENTIAL:
				value = Math.exp(left.getValue());
				break;
		}
	}

	private void accumulateGradient(double grad) {
		this.grad = this.grad + grad;
	}

	private void calcGradient() {
		gradCount++;
		double temp;
		switch (type) {
			case VALUE:
			case LEARNABLE:
				break;
			case ADD:
				left.accumulateGradient(grad);
				right.accumulateGradient(grad);
				break;
			case SUB:
				left.accumulateGradient(grad);
				right.accumulateGradient(-1 * grad);
				break;
			case DIV:
				left.accumulateGradient(grad * (1 / right.getValue()));
				right.accumulateGradient(grad * (left.getValue() * (-1 / (right.getValue() * right.getValue()))));
				break;
			case MUL:
				left.accumulateGradient(grad * right.getValue());
				right.accumulateGradient(grad * left.getValue());
				break;
			case POW:
				temp = right.getValue();
				left.accumulateGradient(grad * temp * Math.pow(left.getValue(), (temp-1)));
				right.accumulateGradient(grad * value * Math.log(left.getValue()));
				break;
			case SIN:
				left.accumulateGradient(grad * Math.cos(left.getValue()));
				break;
			case COS:
				left.accumulateGradient(grad * -1 * Math.sin(left.getValue()));
				break;
			case TAN:
				temp = Math.cos(left.getValue());
				temp = temp * temp;
				left.accumulateGradient(grad / temp);
				break;
			case SINH:
				left.accumulateGradient(grad * Math.cosh(left.getValue()));
				break;
			case COSH:
				left.accumulateGradient(grad * Math.sinh(left.getValue()));
				break;
			case TANH:
				temp = Math.cosh(left.getValue());
				temp = temp * temp;
				left.accumulateGradient(grad / temp);
				break;
			case RELU:
				temp = (value < 0) ? negSlope : 1;
				left.accumulateGradient(grad * temp);
				break;
			case EXPONENTIAL:
				left.accumulateGradient(grad * value);
				break;
		}
	}

	private void orderValues() {
		if (null == this.values) {
			List<JAutogradValue>tempList = orderValues(new HashSet<String>(), new LinkedList<JAutogradValue>());
			this.values = new JAutogradValue[tempList.size()];
			int i = 0;
			for (JAutogradValue value : tempList) {
				this.values[i++] = value;
			}
		}
	}

	private List<JAutogradValue> orderValues(HashSet<String> trackSet, LinkedList<JAutogradValue> values) {
		if (trackSet.contains(id)) {
			return values;
		}
		trackSet.add(id);
		if (null != left) {
			left.orderValues(trackSet, values);
		}
		if (null != right) {
			right.orderValues(trackSet, values);
		}
		values.add(this);
		return values;
	}

	@Override
	public void forward() {
		orderValues();
		for (int i = 0; i < values.length; i++) {
			values[i].calcValue();
		}
	}

	@Override
	public void backward() {
		orderValues();
		this.grad = 1;
		for (int i = values.length - 1; i >= 0; i--) {
			values[i].calcGradient();
		}
	}

	@Override
	public void learn(double rate) {
		orderValues();
		for (int i = 0; i < values.length; i++) {
			values[i].value = values[i].value - (rate * grad / gradCount);
		}
	}

	@Override
	public void reset() {
		if(null!=values) {
			for (int i = 0; i < values.length; i++) {
				values[i].resetState();
			}
		} else {
			resetState();
		}
	}

	private void resetState() {
		if (ValueType.VALUE != type) {
			value = 0;
		}
		this.grad = 0;
		this.gradCount = 0;
	}

	@Override
	public String toString() {
		DecimalFormat df = new DecimalFormat("#.##");
		return df.format(value);
	}

	@Override
	public void createDotGraph() {
		orderValues();
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		PrintWriter dot = new PrintWriter(baos);
		dot.println("digraph Values {");
		dot.println("\tnode [shape=record];");
		for (int i = 0; i < values.length; i++) {
			dot.println("\t" + values[i].createLabel(i));
		}
		dot.println("}");
	}

	private String createLabel(int index) {
		String label = "";
		switch(type) {
			case VALUE:
				label = "v"+index+" [label=\"{v"+index+"|{<v>"+value+"|"+grad+"| "+(long)gradCount+"}}\"];";
				break;
			case LEARNABLE:
				label = "l"+index+" [label=\"{l"+index+"|{<v>"+value+"|"+grad+"| "+(long)gradCount+"}}\"];";
				break;
			case ADD:
				label = "o"+index+" [label=\"{<op>+|{"+value+"|"+grad+"|"+gradCount+"}}\"];";
				break;
			case SUB:
				label = "o"+index+" [label=\"{<op>-|{"+value+"|"+grad+"|"+gradCount+"}}\"];";
				break;
			case DIV:
				label = "o"+index+" [label=\"{<op>/|{"+value+"|"+grad+"|"+gradCount+"}}\"];";
				break;
			case MUL:
				label = "o"+index+" [label=\"{<op>*|{"+value+"|"+grad+"|"+gradCount+"}}\"];";
				break;
			case POW:
				label = "o"+index+" [label=\"{<op>^|{"+value+"|"+grad+"|"+gradCount+"}}\"];";
				break;
			case SIN:
				label = "o"+index+" [label=\"{<op>sin|{"+value+"|"+grad+"|"+gradCount+"}}\"];";
				break;
			case COS:
				label = "o"+index+" [label=\"{<op>cos|{"+value+"|"+grad+"|"+gradCount+"}}\"];";
				break;
			case TAN:
				label = "o"+index+" [label=\"{<op>tan|{"+value+"|"+grad+"|"+gradCount+"}}\"];";
				break;
			case SINH:
				label = "o"+index+" [label=\"{<op>sinh|{"+value+"|"+grad+"|"+gradCount+"}}\"];";
				break;
			case COSH:
				label = "o"+index+" [label=\"{<op>cosh|{"+value+"|"+grad+"|"+gradCount+"}}\"];";
				break;
			case TANH:
				label = "o"+index+" [label=\"{<op>tanh|{"+value+"|"+grad+"|"+gradCount+"}}\"];";
				break;
			case RELU:
				label = "o"+index+" [label=\"{<op>relu|{"+value+"|"+grad+"|"+gradCount+"}}\"];";
				break;
			case EXPONENTIAL:
				label = "o"+index+" [label=\"{<op>e^x|{"+value+"|"+grad+"|"+gradCount+"}}\"];";
				break;
		}
		return label;
	}
}

/*
digraph G {
  node [shape=record];
v2 [label="{<op>+|{3|1}}"];
v0 [label="{v0|{<v>1|1}}"];
v1 [label="{v1|{<v>2|1}}"];

v0:v -> v2:op
v1:v -> v2:op
}

*/

