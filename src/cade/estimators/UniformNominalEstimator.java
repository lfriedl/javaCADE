package cade.estimators;

import java.util.HashSet;

public class UniformNominalEstimator extends Estimator {
    public HashSet<Integer> valuesSeen;
    public UniformNominalEstimator(HashSet<Integer> valuesSeen) {
       	this.valuesSeen = valuesSeen;
    }

    @Override
    public double probabilityOf(double attrValue) {
        if (attrValue == (int) attrValue && valuesSeen.contains((int)attrValue)) {
            return 1 / (double) valuesSeen.size();
        } else {
            return 0;
        }
    }
}
