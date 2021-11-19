package cade.estimators;

import weka.core.Instances;
import java.util.Arrays;

/**
 * An estimated probability distribution for a single attribute.
 */
public abstract class Estimator {
    // Child classes should implement one of the two probabilityOf() methods.
    // Either the little scalar version (if it's efficient)--and we'll loop over it--or a better batch/array version
    // (in which case the scalar version never gets called).
    public double probabilityOf(double attrValue) { return -1; };  // Cause errors if this base class's method is ever called

    // Usually can make the calls one at a time. Only for R, in KDEstimator, is this different.
    public double[] probabilityOf(double[] attrVals) {
        double[] results = new double[attrVals.length];
        for (int i = 0; i < attrVals.length; i++) {
            results[i] = probabilityOf(attrVals[i]);
        }
        return results;
    }

    protected double[] valuesWithoutMissing(Instances instances, int attrNum) {
        double[] valuesTmp = new double[instances.numInstances()];
        int numEntries = 0;
        for (int i = 0; i < instances.numInstances(); i++) {
            if (! instances.instance(i).isMissing(attrNum)) {
                valuesTmp[numEntries] = instances.instance(i).value(attrNum);
                numEntries++;
            }
        }
        return Arrays.copyOf(valuesTmp, numEntries);
    }

}
