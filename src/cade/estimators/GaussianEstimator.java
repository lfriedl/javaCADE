package cade.estimators;

import java.util.Random;

public class GaussianEstimator extends Estimator {
	public double mean;
	public double stdDev;
    protected Random randomness;

    public GaussianEstimator(double mean, double stdDev, Random rng) {
        super();
        this.mean = mean;
        this.stdDev = stdDev;
        this.randomness = new Random(rng.nextInt());
    }

    @Override
    // Lines copied from http://weka.sourceforge.net/doc.packages/oneClassClassifier --> GaussianGenerator.java
    // (instead of requiring that package as a dependency).
    public double probabilityOf(double attrValue) {
        double twopisqrt = Math.sqrt(2 * Math.PI);
        double left = 1 / (stdDev * twopisqrt);
        double diffsquared = Math.pow((attrValue - mean), 2);
        double bottomright = 2 * Math.pow(stdDev, 2);
        double brackets = -1 * (diffsquared / bottomright);

        double probx = left * Math.exp(brackets);

        return probx;
    }

    public double generate() {
        double gaussian = randomness.nextGaussian();
        return(mean + (gaussian * stdDev));
    }
}
