package com.ashutoshwad.utils.jautograd.test;

import com.ashutoshwad.utils.jautograd.ExecutorFactory;
import com.ashutoshwad.utils.jautograd.Matrix;
import org.junit.jupiter.api.Test;

public class ViewTest {
    @Test
    public void explodeTest() {
        ExecutorFactory.createExecutor(1);
        Matrix x = Matrix.createXavierGlorotInitializedMatrix(8096, 512, true);
        Matrix w = Matrix.createXavierGlorotInitializedMatrix(512, 64, true);
        Matrix c = x.matmul(w).swish();
        long start = 0;
        long end = 0;
        start = System.currentTimeMillis();
        c.forward();
        end = System.currentTimeMillis();
        System.out.println("forward: " + (end - start));
        start = System.currentTimeMillis();
        c.backward();
        end = System.currentTimeMillis();
        System.out.println("backward: " + (end - start));
    }
}
