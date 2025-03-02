package com.ashutoshwad.utils.jautograd.compute;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import static org.junit.Assert.assertEquals;

@RunWith(JUnit4.class)
public class ComputeNodeTest {
    @Test
    public void testCalc() {
        ComputeNode c1 = new ComputeNode(1);
        ComputeNode c2 = new ComputeNode(2);
        ComputeNode c3 = c1.mul(c2);
        assertEquals(2, c3.getValue(), 0.01);
        c3.computeGradient();
        c2.computeGradient();
        c1.computeGradient();
        System.out.println(c1.getGradient());
    }
    @Test
    public void visitorTest() {
        ComputeNodeVisitor visitor = new ComputeNodeVisitor();
        ComputeNode c1 = new ComputeNode(1);
        ComputeNode c2 = new ComputeNode(2);
        ComputeNode c3 = c1.mul(c2);
        c1.visit(visitor);
        visitor.prepareBatches();
        System.out.println("Done");
    }
}
