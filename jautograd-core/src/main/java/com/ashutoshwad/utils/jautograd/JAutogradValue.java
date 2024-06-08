package com.ashutoshwad.utils.jautograd;

import java.io.ByteArrayOutputStream;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

import com.ashutoshwad.utils.jautograd.exception.JAutogradException;

class JAutogradValue implements Value {
	public static enum ValueType {
		VALUE, LEARNABLE, ADD, SUB, DIV, MUL, POW, SIN, COS, TAN, SINH, COSH, TANH, RELU, EXPONENTIAL, NATURAL_LOG
	}

	private final String id;
	private final JAutogradValue left;
	private final JAutogradValue right;
	private final ValueType type;
	private final double negSlope;
	private double value;
	private double tempGradient;
	private double gradient;
	private int gradientCount;
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
		this.gradient = 0;
		this.tempGradient = 0;
		this.gradientCount = 0;
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
		return gradient;
	}

	public void setGradient(double gradient) {
		this.gradient = gradient;
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
	public Value sqrt() {
		return pow(new JAutogradValue(0.5));
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

	@Override
	public Value ln() {
		return new JAutogradValue(this, null, ValueType.NATURAL_LOG);
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
			case NATURAL_LOG:
				value = Math.log(left.getValue());
				break;
		}
	}

	private void accumulateGradient(double gradient) {
		this.tempGradient +=  gradient;
	}

	private void calcGradient() {
		double temp;
		switch (type) {
			case VALUE:
			case LEARNABLE:
				break;
			case ADD:
				left.accumulateGradient(tempGradient);
				right.accumulateGradient(tempGradient);
				break;
			case SUB:
				left.accumulateGradient(tempGradient);
				right.accumulateGradient(-1 * tempGradient);
				break;
			case DIV:
				left.accumulateGradient(tempGradient * (1 / right.getValue()));
				right.accumulateGradient(tempGradient * (left.getValue() * (-1 / (right.getValue() * right.getValue()))));
				break;
			case MUL:
				left.accumulateGradient(tempGradient * right.getValue());
				right.accumulateGradient(tempGradient * left.getValue());
				break;
			case POW:
				temp = right.getValue();
				left.accumulateGradient(tempGradient * temp * Math.pow(left.getValue(), (temp-1)));
				right.accumulateGradient(tempGradient * value * Math.log(left.getValue()));
				break;
			case SIN:
				left.accumulateGradient(tempGradient * Math.cos(left.getValue()));
				break;
			case COS:
				left.accumulateGradient(tempGradient * -1 * Math.sin(left.getValue()));
				break;
			case TAN:
				temp = Math.cos(left.getValue());
				temp = temp * temp;
				left.accumulateGradient(tempGradient / temp);
				break;
			case SINH:
				left.accumulateGradient(tempGradient * Math.cosh(left.getValue()));
				break;
			case COSH:
				left.accumulateGradient(tempGradient * Math.sinh(left.getValue()));
				break;
			case TANH:
				temp = Math.cosh(left.getValue());
				temp = temp * temp;
				left.accumulateGradient(tempGradient / temp);
				break;
			case RELU:
				temp = left.getValue();
				temp = (temp == -0.0)?0.0: temp;
				left.accumulateGradient(tempGradient * ((temp >= 0) ? 1 : negSlope));
				break;
			case EXPONENTIAL:
				left.accumulateGradient(tempGradient * value);
				break;
			case NATURAL_LOG:
				left.accumulateGradient(tempGradient * (1/left.getValue()));
				break;
		}
		gradient += tempGradient;
		tempGradient = 0;
		gradientCount++;
	}

	private void orderValues() {
		if (null != this.values) {
			return;
		}
		List<JAutogradValue>tempList = orderValues(new HashSet<String>(), new LinkedList<JAutogradValue>());
		this.values = new JAutogradValue[tempList.size()];
		int i = 0;
		for (JAutogradValue value : tempList) {
			this.values[i++] = value;
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
		this.tempGradient = 1;
		for (int i = values.length - 1; i >= 0; i--) {
			values[i].calcGradient();
		}
	}

	@Override
	public void learn(double rate) {
		orderValues();
		for (int i = 0; i < values.length; i++) {
			values[i].learnUsingGradient(rate);
		}
	}

	public void learnUsingGradient(double rate) {
		if(type != ValueType.LEARNABLE) {
			return;
		}
		if(0 == gradientCount) {
			return;
		}
		double delta = gradient;
		delta = delta * rate;
		delta = delta * -1;
		delta = delta / gradientCount;
		this.value = this.value + delta;
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
		this.tempGradient = 0;
		this.gradient = 0;
		this.gradientCount = 0;
	}

	@Override
	public String toString() {
		DecimalFormat df = new DecimalFormat("#.##");
		return df.format(value);
	}

	@Override
	public String createDotGraph() {
		orderValues();
		Map<String, String> idMap = new HashMap<>();
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		PrintWriter dot = new PrintWriter(baos);
		dot.println("digraph Values {");
		dot.println("\tnode [shape=record];");
		for (int i = 0; i < values.length; i++) {
			dot.println("\t" + values[i].createLabel(i));
			idMap.put(values[i].id, values[i].createLabelName(i));
		}
		createDotMapping(idMap, dot, new HashSet<String>());
		dot.println("}");
		dot.close();
		return new String(baos.toByteArray());
	}
	
	private void createDotMapping(Map<String, String> idMap, PrintWriter dot, Set<String>visitedSet) {
		if(visitedSet.contains(id)) {
			return;
		}
		visitedSet.add(id);
		if(null!=left) {
			dot.println("\t" + idMap.get(left.id) + "->" + idMap.get(id));
			left.createDotMapping(idMap, dot, visitedSet);
		}
		if(null!=right) {
			dot.println("\t" + idMap.get(right.id) + "->" + idMap.get(id));
			right.createDotMapping(idMap, dot, visitedSet);
		}
	}

	private String createLabelName(int index) {
		String labelName = "";
		switch(type) {
			case VALUE:
				labelName = "v"+index;
				break;
			case LEARNABLE:
				labelName = "l"+index;
				break;
			default:
				labelName = "o"+index;
		}
		return labelName;
	}

	private String createLabel(int index) {
		String labelName = createLabelName(index);
		String label = "";
		switch(type) {
			case VALUE:
				label = labelName+" [label=\"{v"+index+"|{Value\\n"+value+"|Gradient\\n"+gradient+"| "+(long)gradientCount+"}}\"];";
				break;
			case LEARNABLE:
				label = labelName+" [label=\"{l"+index+"|{Value\\n"+value+"|Gradient\\n"+gradient+"| "+(long)gradientCount+"}}\"];";
				break;
			case ADD:
				label = labelName+" [label=\"{<op>+|{Value\\n"+value+"|Gradient\\n"+gradient+"|"+gradientCount+"}}\"];";
				break;
			case SUB:
				label = labelName+" [label=\"{<op>-|{Value\\n"+value+"|Gradient\\n"+gradient+"|"+gradientCount+"}}\"];";
				break;
			case DIV:
				label = labelName+" [label=\"{<op>/|{Value\\n"+value+"|Gradient\\n"+gradient+"|"+gradientCount+"}}\"];";
				break;
			case MUL:
				label = labelName+" [label=\"{<op>*|{Value\\n"+value+"|Gradient\\n"+gradient+"|"+gradientCount+"}}\"];";
				break;
			case POW:
				label = labelName+" [label=\"{<op>^|{Value\\n"+value+"|Gradient\\n"+gradient+"|"+gradientCount+"}}\"];";
				break;
			case SIN:
				label = labelName+" [label=\"{<op>sin|{Value\\n"+value+"|Gradient\\n"+gradient+"|"+gradientCount+"}}\"];";
				break;
			case COS:
				label = labelName+" [label=\"{<op>cos|{Value\\n"+value+"|Gradient\\n"+gradient+"|"+gradientCount+"}}\"];";
				break;
			case TAN:
				label = labelName+" [label=\"{<op>tan|{Value\\n"+value+"|Gradient\\n"+gradient+"|"+gradientCount+"}}\"];";
				break;
			case SINH:
				label = labelName+" [label=\"{<op>sinh|{Value\\n"+value+"|Gradient\\n"+gradient+"|"+gradientCount+"}}\"];";
				break;
			case COSH:
				label = labelName+" [label=\"{<op>cosh|{Value\\n"+value+"|Gradient\\n"+gradient+"|"+gradientCount+"}}\"];";
				break;
			case TANH:
				label = labelName+" [label=\"{<op>tanh|{Value\\n"+value+"|Gradient\\n"+gradient+"|"+gradientCount+"}}\"];";
				break;
			case RELU:
				label = labelName+" [label=\"{<op>relu|{Value\\n"+value+"|Gradient\\n"+gradient+"|"+gradientCount+"}}\"];";
				break;
			case EXPONENTIAL:
				label = labelName+" [label=\"{<op>e^x|{Value\\n"+value+"|Gradient\\n"+gradient+"|"+gradientCount+"}}\"];";
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

