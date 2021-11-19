package cade.generativeProbDensities;

import cade.Parameters;
import weka.core.Instances;
import cade.estimators.Estimator;
import cade.estimators.OneEverywhereEstimator;


public class OneEverywhereProbDensity extends ProbDensity {

    public OneEverywhereProbDensity(Parameters params) {
        super(params);
    }

    public Instances generateItems(int numInstances) throws Exception {
        throw new Exception("What?!  You can't use me as a pNeg generator!");
    }

    public Estimator[] constructEstimators() {
        Estimator[] estimators = new Estimator[dataGen.numAttributes];
        for (int i = 0; i < estimators.length; i++)
            estimators[i] = new OneEverywhereEstimator();
        return estimators;
    }

    public String getName() {
        return ("One Everywhere");
    }
}
