package com.ashutoshwad.utils.jautograd.test;

import com.ashutoshwad.utils.jautograd.ExecutorFactory;
import com.ashutoshwad.utils.jautograd.Matrix;
import com.ashutoshwad.utils.jautograd.optimizer.AdamOptimizer;
import com.ashutoshwad.utils.jautograd.optimizer.AdamWOptimizer;
import com.ashutoshwad.utils.jautograd.optimizer.GradientClipper;
import com.ashutoshwad.utils.jautograd.optimizer.StocasticGradientDescentOptimizer;
import org.junit.jupiter.api.Test;

import java.util.LinkedList;
import java.util.Queue;

public class XORTest {
    private Matrix input;
    private Matrix target;
    private Matrix hiddenWeights;
    private Matrix hiddenBias;
    private Matrix outputWeights;
    private Matrix outputBias;

    private AdamWOptimizer optimizer;
    private GradientClipper clipper;

    @Test
    public void testXor() {
        ExecutorFactory.createExecutor(1);
        input = createXORInput();
        target = createXORTarget();

        // Initialize network parameters
        this.hiddenWeights = Matrix.createXavierGlorotInitializedMatrix(2, 8, true);  // 2 -> 8 hidden
        this.hiddenBias = Matrix.create(1, 8, () -> 0.0, true);
        this.outputWeights = Matrix.createXavierGlorotInitializedMatrix(8, 1, true);  // 8 -> 1 output
        this.outputBias = Matrix.create(1, 1, () -> 0.0, true);

        // Setup optimizer and gradient clipper
        this.optimizer = new AdamWOptimizer(0.01);  // Learning rate
        this.clipper = new GradientClipper(0.01);

        // Add all parameters to optimizer and clipper
        addParametersToOptimizer();

        //Create graph
        Matrix intermediate = input.matmul(hiddenWeights).add(hiddenBias).swish();
        Matrix output = intermediate.matmul(outputWeights).add(outputBias).sigmoid();

        Matrix loss_1 = output.sub(target);
        Matrix loss = mse(loss_1);
        loss.backward();

        train(1000, output, target, loss);
    }

    private void printMatrix(String name, Matrix weights) {
        System.out.println("Matrix: " + name);
        System.out.println("Values: ");
        System.out.println(weights.getPrintableMatrixValues());
        System.out.println("Gradients: ");
        System.out.println(weights.getPrintableMatrixGradients());
    }

    public void train(int epochs, Matrix prediction, Matrix target, Matrix loss) {
        System.out.println("Starting XOR training with your Matrix framework...");
        System.out.println("Epoch\tLoss\t\tPredictions");
        System.out.println("-----\t----\t\t-----------");

        for (int epoch = 0; epoch < epochs; epoch++) {
            // Forward pass
            loss.forward();

            // Zero gradients
            loss.zeroGrad();

            // Backward pass - compute MSE gradient manually
            loss.backward();

            // Clip gradients
            clipper.clipGradients(1.0);

            // Update parameters
            optimizer.learn();

            // Print progress every 200 epochs
            if (epoch % 10 == 0 || epoch == epochs - 1) {
                System.out.printf("%d\t%.6f\t[%.3f, %.3f, %.3f, %.3f]\n",
                        epoch, loss.getValue(),
                        prediction.getValue(0, 0),
                        prediction.getValue(1, 0),
                        prediction.getValue(2, 0),
                        prediction.getValue(3, 0));

                System.out.println("Prediction");
                System.out.println(prediction.getPrintableMatrixValues());
            }
        }
    }

    public void validate(Matrix prediction, Matrix loss) {
        System.out.println("\n=== Final Validation ===");
        loss.forward();

        System.out.println("Input\t\tTarget\tPrediction\tRounded");
        System.out.println("-----\t\t------\t----------\t-------");

        boolean allCorrect = true;
        for (int i = 0; i < 4; i++) {
            double pred = prediction.getValue(i, 0);
            double rounded = Math.round(pred);
            double expected = target.getValue(i, 0);

            System.out.printf("[%.0f, %.0f]\t\t%.0f\t%.6f\t%.0f %s\n",
                    input.getValue(i, 0),
                    input.getValue(i, 1),
                    expected,
                    pred,
                    rounded,
                    (Math.abs(rounded - expected) < 0.1) ? "✓" : "✗");

            if (Math.abs(rounded - expected) >= 0.1) {
                allCorrect = false;
            }
        }

        System.out.println("\nResult: " + (allCorrect ? "🎉 PASSED! Your framework learned XOR!" : "❌ Failed to learn XOR"));
    }

    private Matrix mse(Matrix loss) {
        loss = loss.mul(loss);
        Matrix[][]temp = loss.explode();
        Queue<Matrix>listToAdd = new LinkedList<>();
        for (int row = 0; row < temp.length; row++) {
            for (int col = 0; col < temp[row].length; col++) {
                listToAdd.add(temp[row][col]);
            }
        }
        while(listToAdd.size()>1) {
            Matrix a = listToAdd.poll();
            Matrix b = listToAdd.poll();
            listToAdd.add(a.add(b));
        }
        loss = listToAdd.poll();
        //loss = loss.sqrt();
        return loss;
    }

    private Matrix createXORInput() {
        double[][] data = {
                {0.0, 0.0},  // XOR input 1
                {0.0, 1.0},  // XOR input 2
                {1.0, 0.0},  // XOR input 3
                {1.0, 1.0}   // XOR input 4
        };

        Matrix input = Matrix.create(4, 2);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 2; j++) {
                input.setValue(i, j, data[i][j]);
            }
        }
        Matrix[][]explode = input.explode();
        Matrix a1 = explode[0][0].concatHorizontal(explode[0][1]);
        Matrix a2 = explode[1][0].concatHorizontal(explode[1][1]);
        Matrix a3 = explode[2][0].concatHorizontal(explode[2][1]);
        Matrix a4 = explode[3][0].concatHorizontal(explode[3][1]);
        input = a1.concatVertical(a2).concatVertical(a3).concatVertical(a4);
        return input;
    }

    private Matrix createXORTarget() {
        double[][] data = {
                {0.0},  // 0 XOR 0 = 0
                {1.0},  // 0 XOR 1 = 1
                {1.0},  // 1 XOR 0 = 1
                {0.0}   // 1 XOR 1 = 0
        };

        Matrix target = Matrix.create(4, 1);
        for (int i = 0; i < 4; i++) {
            target.setValue(i, 0, data[i][0]);
        }
        return target;
    }

    private void addParametersToOptimizer() {
        optimizer.addParameter(hiddenWeights);
        optimizer.addParameter(hiddenBias);
        optimizer.addParameter(outputWeights);
        optimizer.addParameter(outputBias);

        clipper.addParameter(hiddenWeights);
        clipper.addParameter(hiddenBias);
        clipper.addParameter(outputWeights);
        clipper.addParameter(outputBias);
    }
}
