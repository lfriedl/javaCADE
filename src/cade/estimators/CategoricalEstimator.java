package cade.estimators;

import weka.core.Instances;

/**
 * Note: Weka will give us "doubles" for the attribute values; these are really
 * just the indices of the categorical values in a lookup table. We never need to look
 * them up; just treat them as indices.
 */
public class CategoricalEstimator extends cade.estimators.Estimator {
    public int[] countsForValue;
    protected int totalCount;

    // Smoothing is always "off" in this class; it's handled elsewhere.
//    public boolean useSmoothing = false;

    public CategoricalEstimator(Instances instances, int attrNum) {
        buildEstimator(instances, attrNum);
    }

    // Provided to make testing easier.
    // counts: the attribute value with index i appears counts[i] times.
    // Weka nominal attributes are indexed 0 to x; this method assumes the input came from there.
    public CategoricalEstimator(int[] counts) {
        countsForValue = counts.clone();
        totalCount = 0;
        for (int valCnt : countsForValue) {
            totalCount += valCnt;
        }
    }

    public void buildEstimator(Instances instances, int attrNum) {
        countsForValue = new int[instances.attribute(attrNum).numValues()];
        totalCount = 0;
        double[] values = valuesWithoutMissing(instances, attrNum);

        for (double val : values) {
            countsForValue[(int) val]++;
            totalCount++;
        }
    }

    @Override
    public double probabilityOf(double attrValue) {
//        if (useSmoothing)
//            return getSmoothedProbability(attrValue);
//        else
            return getUnsmoothedProbability(attrValue);

    }

    protected double getUnsmoothedProbability(double attrValue) {
        if (attrValue < countsForValue.length && countsForValue[(int)attrValue] > 0
                && Math.round(attrValue) == attrValue) {
            return (countsForValue[(int)attrValue]) / (double) totalCount ;
        } else {
            return 0;
        }
    }

}
