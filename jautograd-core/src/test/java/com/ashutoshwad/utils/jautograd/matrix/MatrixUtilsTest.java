package com.ashutoshwad.utils.jautograd.matrix;

import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import com.ashutoshwad.utils.jautograd.Value;

@RunWith(JUnit4.class)
public class MatrixUtilsTest {
	private Value identity[][];
	@Before
	public void before() {
		identity = new Value[][] {
			{Value.of(1), Value.of(0), Value.of(0)},
			{Value.of(0), Value.of(1), Value.of(0)},
			{Value.of(0), Value.of(0), Value.of(1)}
		};
	}
	@Test
	public void testUnitMatrixMultipliedByIdentity() {
		Value[][]a = new Value[][] {
			{Value.of(2), Value.of(2), Value.of(2)},
			{Value.of(2), Value.of(2), Value.of(2)},
			{Value.of(2), Value.of(2), Value.of(2)}
		};

		Value[][]c = MatrixUtils.mul(a, identity);
		for (int i = 0; i < c.length; i++) {
			for (int j = 0; j < c[i].length; j++) {
				assertEquals(2, c[i][j].getValue(), 0.00001);
			}
		}
	}
	@Test
	public void testNonSquareMatrixMultipliedByIdentity() {
		Value[][]a = new Value[][] {
			{Value.of(2), Value.of(3), Value.of(4)},
			{Value.of(5), Value.of(6), Value.of(7)}
		};

		Value[][]c = MatrixUtils.mul(a, identity);
		assertEquals(a.length, c.length);
		for (int i = 0; i < c.length; i++) {
			assertEquals(a[i].length, c[i].length);
			for (int j = 0; j < c[i].length; j++) {
				assertEquals(a[i][j].getValue(), c[i][j].getValue(), 0.00001);
			}
		}
	}
	@Test
	public void testTransposeOfIdentityIsIdentity() {
		Value[][]transpose = MatrixUtils.transpose(identity);
		assertEquals(transpose.length, identity.length);
		for (int i = 0; i < identity.length; i++) {
			assertEquals(transpose[i].length, identity[i].length);
			for (int j = 0; j < identity[i].length; j++) {
				assertEquals(transpose[i][j].getValue(), identity[i][j].getValue(), 0.00001);
			}
		}
	}
	@Test
	public void transposeTwiceShouldReturnOriginalMatrix() {
		Value[][]a = new Value[][] {
			{Value.of(2), Value.of(3), Value.of(4)},
			{Value.of(5), Value.of(6), Value.of(7)}
		};
		Value[][]transpose = MatrixUtils.transpose(MatrixUtils.transpose(a));
		assertEquals(transpose.length, a.length);
		for (int i = 0; i < a.length; i++) {
			assertEquals(transpose[i].length, a[i].length);
			for (int j = 0; j < a[i].length; j++) {
				assertEquals(transpose[i][j].getValue(), a[i][j].getValue(), 0.00001);
			}
		}
	}
}
