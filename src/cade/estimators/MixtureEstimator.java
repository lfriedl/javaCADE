package cade.estimators;

public class MixtureEstimator extends Estimator {
    protected Estimator uniformEstimator;
    protected Estimator marginalEstimator;
    protected double percentUniform;

    public MixtureEstimator(Estimator unifEstimator, Estimator margEstimator, double fractionUniform) {
        uniformEstimator = unifEstimator;
        marginalEstimator = margEstimator;
        this.percentUniform = fractionUniform;
    }

     public double[] probabilityOf(double[] attrVals) {
         double[] p_val_from_unif = uniformEstimator.probabilityOf(attrVals);
         double[] p_val_from_marg = marginalEstimator.probabilityOf(attrVals);
         double[] p_val_from_mixture = new double[attrVals.length];
         for (int i = 0; i < attrVals.length; i++) {
            p_val_from_mixture[i] = percentUniform * p_val_from_unif[i] + (1 - percentUniform) * p_val_from_marg[i];
         }
         return p_val_from_mixture;

     }
}
