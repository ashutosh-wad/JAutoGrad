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
    <artifactId>jautograd-compute</artifactId>
    <version>1.0-SNAPSHOT</version>
</dependency>
```

# Usage

Refer to the test case `com.ashutoshwad.utils.jautograd.test.XORTest` that implements a simple network that learns the XOR operation.
