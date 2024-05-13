package com.ashutoshwad.utils.jautograd.matrix;

import static org.junit.Assert.assertEquals;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.Random;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import com.ashutoshwad.utils.jautograd.Value;

@RunWith(JUnit4.class)
public class XorNeuralNetTest {
	private static final double RATE = 0.001;
	private static Value[][] input;
	private static Value output;
	private static Value expected;
	private static Value loss;

	@Before
	public void before() {
		//Initialize the neural network
		Random r = new Random();
		input = MatrixUtils.createMatrix(1, 2);
		Value[][]layer1 = MatrixUtils.createMatrix(2, 3, ()->r.nextDouble()*2-1);
		Value[][]layer2 = MatrixUtils.createMatrix(3, 1, ()->r.nextDouble()*2-1);

		Value[][]intermidiate = MatrixUtils.mul(input, layer1);
		MatrixUtils.op(intermidiate, v -> v.add(Value.learnable(r.nextDouble()*2-1)));
		MatrixUtils.op(intermidiate, v -> v.tanh());
		
		intermidiate = MatrixUtils.mul(intermidiate, layer2);
		MatrixUtils.op(intermidiate, v -> v.add(Value.learnable(r.nextDouble()*2-1)));
		MatrixUtils.op(intermidiate, v -> v.tanh());
		
		output = intermidiate[0][0];
		expected = Value.of(0);
		loss = output.sub(expected);
		loss = loss.mul(loss);
	}

	@Test
	public void learnXor() {
		//Start learning
		double lossAgg = 10;
		while (lossAgg > 0.01) {
			lossAgg = 0;
			forward(0, 0, 0, false);lossAgg += loss.getValue();
			forward(0, 1, 1, false);lossAgg += loss.getValue();
			forward(1, 0, 1, false);lossAgg += loss.getValue();
			forward(1, 1, 0, false);lossAgg += loss.getValue();
			loss.learn(RATE);
			loss.reset();
		}
		System.out.println("X\tY\tOut\tAct\tLoss");
		forward(0, 0, 0, true);
		forward(0, 1, 1, true);
		forward(1, 0, 1, true);
		forward(1, 1, 0, true);

		assertWorking(0, 0, 0);
		assertWorking(0, 1, 1);
		assertWorking(1, 0, 1);
		assertWorking(1, 1, 0);
	}

	public static final void forward(double i1, double i2, double expectedValue, boolean print) {
		input[0][0].setValue(i1);
		input[0][1].setValue(i2);
		expected.setValue(expectedValue);
		loss.forward();
		loss.backward();
		if(!print) {
			return;
		}
		DecimalFormat df = new DecimalFormat("0.0000");
		System.out.print(df.format(input[0][0].getValue()) + "\t");
		System.out.print(df.format(input[0][1].getValue()) + "\t");
		System.out.print(df.format(output.getValue()) + "\t");
		System.out.print(df.format(expected.getValue()) + "\t");
		System.out.println(df.format(loss.getValue()) + "\t");
		//MatrixUtils.print(input);System.out.println();
		//System.out.println(output.getValue());
		System.out.println("------------------------------------------------------------------------");
	}

	public static final void assertWorking(double i1, double i2, double expectedValue) {
		input[0][0].setValue(i1);
		input[0][1].setValue(i2);
		expected.setValue(expectedValue);
		loss.forward();
		double outputVal = new BigDecimal(output.getValue()).setScale(0, RoundingMode.HALF_UP).doubleValue();
		assertEquals(expectedValue, outputVal, 0.001);
	}
}
