# JAutoGrad
JAutoGrad is an autograd library for the Java programming language.
It can be used as a building block towards the making and understanding of further machine learning algorithms.
This library has been inspired by [micrograd](https://github.com/karpathy/micrograd), a library created by Andrej Karpathy.

# Installation

Clone this repository and build it using maven.
```shell
git clone https://github.com/ashutosh-wad/JAutoGrad.git
cd JAutoGrad
mvn clean install
```

Now you just need to include the dependency of this project in your own work.

```xml
<dependency>
    <groupId>com.ashutoshwad.utils.jautograd</groupId>
    <artifactId>JAutoGrad</artifactId>
    <version>0.0.1-SNAPSHOT</version>
</dependency>
```

# Usage

Below is a code snippet implementing a simple network with 2 inputs 3 hidden layers and 1 output which learns the XOR function.
```Java
import java.text.DecimalFormat;

import com.ashutoshwad.utils.jautograd.Value;
import com.ashutoshwad.utils.jautograd.toynet.ToyNetwork;

public class Test {
	private static final double RATE = 0.001;
	private static Value[] input;
	private static Value output;
	private static Value expected;
	private static Value loss;
	public static void main(String[] args) {
		//Create network
		input = new Value[] {Value.of(0), Value.of(0)};
		ToyNetwork network = new ToyNetwork(2, 3, 1);
		output = network.connect(input)[0];
		expected = Value.of(0);
		loss = output.sub(expected);
		loss = loss.mul(loss);

		//Train network
		double lossAgg = 10;
		while (lossAgg > 0.001) {
			lossAgg = 0;
			forward(0, 0, 0, false);
			lossAgg += loss.getValue();
			forward(0, 1, 1, false);
			lossAgg += loss.getValue();
			forward(1, 0, 1, false);
			lossAgg += loss.getValue();
			forward(1, 1, 0, false);
			lossAgg += loss.getValue();
			learn();
		}

		//See results
		System.out.println("Input1\tInput2\tOutput\tExpected\tLoss");
		forward(0, 0, 0, true);
		forward(0, 1, 1, true);
		forward(1, 0, 1, true);
		forward(1, 1, 0, true);
	}

	private static void forward(int i1, int i2, int o, boolean print) {
		input[0].setValue(i1);
		input[1].setValue(i2);
		expected.setValue(o);
		loss.forward();
		loss.backward();
		if(!print) {
			return;
		}
		DecimalFormat df = new DecimalFormat("0.0000");
		System.out.print(df.format(input[0].getValue()));
		System.out.print("\t");
		System.out.print(df.format(input[1].getValue()));
		System.out.print("\t");
		System.out.print(df.format(output.getValue()));
		System.out.print("\t");
		System.out.print(df.format(expected.getValue()));
		System.out.print("\t");
		System.out.print(df.format(loss.getValue()));
		System.out.println();
	}

	private static void learn() {
		loss.learn(RATE);
		loss.reset();
	}
}
```
