package com.ashutoshwad.utils.jautograd.compute;

import static org.junit.Assert.*;
import org.junit.Test;

public class XORNetworTest {
    @Test
    public void xorMatrixTest() {
        Matrix input = createInputMatrix();

        Matrix hidden = Matrix.createXavierGlorotInitializedMatrix(2, 4, true);
        hidden = input.matmul(hidden);
        Matrix hidden_bias = Matrix.createMatrix(1, 4, () -> 0.0, true);
        hidden = hidden.add(hidden_bias);
        hidden = hidden.simpleSwish();

        Matrix out = Matrix.createXavierGlorotInitializedMatrix(4, 1, true);
        out = hidden.matmul(out);
        Matrix out_bias = Matrix.createMatrix(1, 1, () -> 0.0, true);
        out = out.add(out_bias);
        out = out.sigmoid();

        Matrix expected = Matrix.createMatrix(4, 1);
        expected.get(0, 0).setValue(0.0);
        expected.get(1, 0).setValue(1.0);
        expected.get(2, 0).setValue(1.0);
        expected.get(3, 0).setValue(0.0);

        Matrix loss = out.sub(expected);
        loss = loss.op((node) -> node.mul(node));
        loss = loss.sum();
        loss = loss.sqrt();

        JAutogradExecutor executor = loss.get(0, 0).createExecutor(4);

        for (int i = 0; i < 1000; i++) {
            executor.zeroGradAndForward();
            executor.backward();
            executor.op(node -> {
                if(node.isTrainable()) {
                    node.setValue(node.getValue() + 0.1 * (node.getGradient() * -1));
                }
            });
        }

        System.out.println(input);
        System.out.println(out);
        assertEquals(0, Math.round(out.get(0, 0).getValue()));
        assertEquals(1, Math.round(out.get(1, 0).getValue()));
        assertEquals(1, Math.round(out.get(2, 0).getValue()));
        assertEquals(0, Math.round(out.get(3, 0).getValue()));

    }

    private Matrix createInputMatrix() {
        Matrix input = Matrix.createMatrix(4, 2);
        input.get(0, 0).setValue(0);
        input.get(0, 1).setValue(0);

        input.get(1, 0).setValue(0);
        input.get(1, 1).setValue(1);

        input.get(2, 0).setValue(1);
        input.get(2, 1).setValue(0);

        input.get(3, 0).setValue(1);
        input.get(3, 1).setValue(1);
        return input;
    }
}
