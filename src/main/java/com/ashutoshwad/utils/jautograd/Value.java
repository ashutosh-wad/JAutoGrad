package com.ashutoshwad.utils.jautograd;

import java.io.ByteArrayOutputStream;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;

public class Value {
	private static final AtomicLong ID_FACTORY = new AtomicLong(0);

	public static enum Type {
		VALUE, ADD, SUB, DIV, MUL, TANH, RELU, FUNC
	}

	private final long id;
	private double value;
	private double grad;
	private List<Value> children;
	private Function<Double, Double> function;
	private Type type;

	public Value(double value) {
		this(value, new LinkedList<>(), Type.VALUE);
	}

	private Value(double value, List<Value> children, Type type) {
		this.value = value;
		this.children = children;
		this.type = type;
		this.grad = 0;
		this.id = ID_FACTORY.getAndIncrement();
	}

	public long getId() {
		return id;
	}

	public void initGrad() {
		this.grad = 0;
	}

	public double getValue() {
		return value;
	}

	public double getGrad() {
		return grad;
	}

	public Value add(Value n) {
		Value retVal = new Value(value + n.value, toList(this, n), Type.ADD);
		return retVal;
	}

	public Value sub(Value n) {
		Value retVal = new Value(value - n.value, toList(this, n), Type.SUB);
		return retVal;
	}

	public Value mul(Value n) {
		Value retVal = new Value(value * n.value, toList(this, n), Type.MUL);
		return retVal;
	}

	public Value div(Value n) {
		Value retVal = new Value(value / n.value, toList(this, n), Type.DIV);
		return retVal;
	}

	public Value tanh() {
		Value retVal = new Value(Math.tanh(value), toList(this), Type.TANH);
		return retVal;
	}

	public Value relu() {
		Value retVal = new Value((value <= 0) ? 0 : value, toList(this), Type.RELU);
		return retVal;
	}

	public Value function(Function<Double, Double> function) {
		Value retVal = new Value(function.apply(value), toList(this), Type.FUNC);
		retVal.function = function;
		return retVal;
	}

	public void back() {
		switch (type) {
			case ADD:
				backAdd();
				break;
			case SUB:
				backSub();
				break;
			case MUL:
				backMul();
				break;
			case DIV:
				backDiv();
				break;
			case TANH:
				backTanh();
				break;
			case RELU:
				backRelu();
				break;
			case FUNC:
				backFunc();
				break;
		}
	}

	private void accumulateGrad(double grad) {
		this.grad += grad;
	}

	private void backAdd() {
		for (Value node : children) {
			node.accumulateGrad(this.grad);
		}
	}

	private void backSub() {
		children.get(0).accumulateGrad(grad);
		children.get(1).accumulateGrad(-1 * grad);
	}

	private void backMul() {
		children.get(0).accumulateGrad(children.get(1).value * grad);
		children.get(1).accumulateGrad(children.get(0).value * grad);
	}

	private void backDiv() {
		double a = children.get(0).value;
		double b = children.get(1).value;
		children.get(0).accumulateGrad((1 / b) * this.grad);
		children.get(1).accumulateGrad((a * (-1 / (b * b))) * this.grad);
	}

	private void backTanh() {
		double dtanh = 1 - (value * value);
		children.get(0).accumulateGrad(dtanh * this.grad);
	}

	private void backRelu() {
		double v = children.get(0).value;
		if (v <= 0) {
			children.get(0).accumulateGrad(0);
		} else {
			children.get(0).accumulateGrad(this.grad);
		}
	}

	private void backFunc() {
		double h = 0.00001;
		double a = children.get(0).value;
		double b = a + h;
		double delta = (function.apply(b) - function.apply(a)) / h;
		children.get(0).accumulateGrad(delta * this.grad);
	}

	private List<Value> toList(Value... nodes) {
		List<Value> nodeList = new LinkedList<Value>();
		if (null == nodes) {
			return nodeList;
		}
		if (0 == nodes.length) {
			return nodeList;
		}
		for (Value node : nodes) {
			nodeList.add(node);
		}
		return nodeList;
	}

	@Override
	public String toString() {
		return "Node [value=" + value + ", grad=" + grad + ", type=" + type + "]";
	}

	public List<Value> readChildren() {
		List<Value> nodes = new LinkedList<>();
		nodes.addAll(children);
		return nodes;
	}

	public void backPropogate() {
		Queue<Value> queue = orderNodes();
		grad = 1;
		while(queue.size()>0) {
			queue.poll().back();
		}
	}

	private Queue<Value> orderNodes() {
		Map<Long, Set<Long>> parentToChild = new HashMap<>();
		Map<Long, Set<Long>> childToParent = new HashMap<>();
		Map<Long, Value> nodeMap = new HashMap<>();
		collectAllNodes(parentToChild, childToParent, nodeMap, this);
		Queue<Value> nodeQueue = new LinkedList<>();
		Queue<Value> orderedNodeQueue = new LinkedList<>();
		nodeQueue.addAll(nodeMap.values());
		while (nodeQueue.size() > 0) {
			Value node = nodeQueue.poll();
			if (childToParent.containsKey(node.getId())) {
				//This indicates at least one parent exists so we do not continue.
				nodeQueue.add(node);
			} else {
				node.initGrad();
				orderedNodeQueue.add(node);
				long parentId = node.getId();
				if(parentToChild.containsKey(parentId)) {
					Set<Long>childSet = parentToChild.get(parentId);
					for (Long childId : childSet) {
						Set<Long> parentSet = childToParent.get(childId);
						parentSet.remove(parentId);
						if(parentSet.size()==0) {
							childToParent.remove(childId);
						}
					}
					childSet.clear();
					parentToChild.remove(parentId);
				}
			}
		}
		return orderedNodeQueue;
	}

	private void collectAllNodes(Map<Long, Set<Long>> parentToChild, Map<Long, Set<Long>> childToParent,
			Map<Long, Value> nodeMap, Value node) {
		long id = node.getId();
		nodeMap.put(id, node);
		if (!parentToChild.containsKey(id)) {
			parentToChild.put(id, new HashSet<>());
		}
		Set<Long> childSet = new HashSet<>();
		for (Value n : node.children) {
			parentToChild.get(id).add(n.getId());
			if (!childToParent.containsKey(n.getId())) {
				childToParent.put(n.getId(), new HashSet<>());
			}
			childToParent.get(n.getId()).add(id);
			collectAllNodes(parentToChild, childToParent, nodeMap, n);
		}
	}

	public String getDot() {
		StringBuilder sb = new StringBuilder();
		sb.append("digraph G {\n");
		sb.append(getDotContent());
		sb.append("\n}");
		return sb.toString();
	}

	public String getDotContent() {
		String nodeDot = "n" + id + " [label=\"" + String.format("data %.4f | grad %.4f | "+type, value, grad) + "\" shape=record]";
		if (type == Type.VALUE) {
			return nodeDot;
		} else {
			ByteArrayOutputStream baos = new ByteArrayOutputStream();
			PrintWriter pw = new PrintWriter(baos);
			pw.println(nodeDot);
			String opLab = "";
			if (type == Type.ADD) {
				printDotDualOp("+", pw);
			} else if (type == Type.SUB) {
				printDotDualOp("-", pw);
			} else if (type == Type.MUL) {
				printDotDualOp("X", pw);
			} else if (type == Type.DIV) {
				printDotDualOp("/", pw);
			} else if (type == Type.TANH) {
				printDotSingleOp("tanh()", pw);
			} else if (type == Type.RELU) {
				printDotSingleOp("relu()", pw);
			} else if (type == Type.FUNC) {
				printDotSingleOp("func()", pw);
			} else {
				System.out.println();
			}
			pw.close();
			return new String(baos.toByteArray());
		}
	}

	private void printDotSingleOp(String opLabel, PrintWriter pw) {
		long opNodeId = ID_FACTORY.getAndIncrement();
		String opNode = "n" + opNodeId + " [label=\"" + opLabel + "\" shape=circle]";
		String c1 = children.get(0).getDotContent();
		pw.println(opNode);
		pw.println(c1);
		pw.println("n" + opNodeId + "->n" + id);
		pw.println("n" + children.get(0).getId() + "->n" + opNodeId);
	}

	private void printDotDualOp(String opLabel, PrintWriter pw) {
		long opNodeId = ID_FACTORY.getAndIncrement();
		String opNode = "n" + opNodeId + " [label=\"" + opLabel + "\" shape=circle]";
		String c1 = children.get(0).getDotContent();
		String c2 = children.get(1).getDotContent();
		pw.println(opNode);
		pw.println(c1);
		pw.println(c2);
		pw.println("n" + opNodeId + "->n" + id);
		pw.println("n" + children.get(0).getId() + "->n" + opNodeId);
		pw.println("n" + children.get(1).getId() + "->n" + opNodeId);
	}
}
