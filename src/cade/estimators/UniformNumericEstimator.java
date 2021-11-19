package cade.estimators;

public class UniformNumericEstimator extends Estimator {
    public double min, max;
    public UniformNumericEstimator(double min, double max) {
        super();
        this.min = min;
        this.max = max;
    }

    public double probabilityOf(double attrValue) {
        if (attrValue >= min && attrValue <= max) {
            if (max != min)
                return 1 / (max - min);
            else    // weird case: if it's a point spike. It's like having a categorical estimator, with just 1 value.
                return 1;
        }
        else
            return 0;
    }
}
