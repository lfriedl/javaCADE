package cade.generativeProbDensities;

import java.util.Random;

import cade.Parameters;
import cade.positiveDataGenerators.DataGenerator;
import weka.core.Instance;
import weka.core.Instances;
import cade.estimators.Estimator;


public abstract class ProbDensity implements GenerativeProbDensity {
    /**
     * For GenerativeProbDensity, this base class must be able to:
     * -getDataGenerator() and generateItems(), and computeLogDensity() of new data instances
     *
     * Order that things will be called:
     * -positive DataGenerator sets up training data
     * -we construct a ProbDensity, and immediately generateItems() from it
     * -then much later, call ProbDensity.computeLogDensity() of new data.
     *
     * The idea is that each ProbDensity stores 1 or more Estimators that learn/fit a density to training data.
     * The Estimators can be created at initialization (and saved) OR any time after. They will be asked for
     * (and re-saved) by computeLogDensity().
     */
    protected DataGenerator dataGen;
    protected Random rng;
    protected boolean useSmoothing;
    protected double newValToUseSmoothing = 0;
    protected Estimator[] estimatorsForAttributes;
    protected int sizeAtWhichToSubdivide = 10000;
    protected int partitionSize = 1000;

    protected boolean printInfinityDebugInfo = false;
    
    public abstract Instances generateItems(int numInstances) throws Exception;

    public ProbDensity(Parameters params) {
        dataGen = params.getDataGenerator();
        useSmoothing = params.smoothToRemove1sAnd0s;
        rng = new Random();
    }
    
    public ProbDensity(DataGenerator dataGen){
    	this.dataGen = dataGen;
    	useSmoothing = true;
    	rng = new Random();
    }
    
    public ProbDensity(){
    	this.dataGen = null;
    	useSmoothing = true;
    	rng = new Random();
    }

    public void setClassLabelNegative(Instances instances) {
        for (int i = 0; i < instances.numInstances(); i++) {
            instances.instance(i).setClassValue(Parameters.negativeClassLabel);
        }
    }

    public void setRandom(Random r) {
        rng = r;
    }

    // N.B. Make sure these estimators handle missing values appropriately; they're present in the data.
    public Estimator[] getEstimatorsForAttributes() {
        if (estimatorsForAttributes == null) {
            estimatorsForAttributes = constructEstimators();
        }
        return estimatorsForAttributes;
    }

    public abstract Estimator[] constructEstimators();

    public double[] computeLogDensity(Instances testInstances) {
        estimatorsForAttributes = getEstimatorsForAttributes();
        // These arrays range along the testInstances
        double[] currentLogProbs = new double[testInstances.numInstances()];  // i.e., currentProb = e^currentLogProbs

        // Need to subdivide the calls to R, or else they take too long.
        // Prefer to subdivide as close to the actual calls as possible.
        boolean goingToSubdivide = false;
        Instances[] smallTestInstancesArray = new Instances[0];
        int numPartitions = 0;
        // Only subdivide if we're using KDE pNegs, which involve calls to R
        if (testInstances.numInstances() > sizeAtWhichToSubdivide && this instanceof IndepMarginalsProbDensity) {
            goingToSubdivide = true;
        }

        if (goingToSubdivide) {
            //          System.out.println("getting probability estimates for " + testInstances.numInstances() +
            //                 " instances, " + partitionSize + " at a time");
            // Set up an array of smallTestInstances
            numPartitions = (int) Math.ceil(testInstances.numInstances() / (double) partitionSize);
            smallTestInstancesArray = new Instances[numPartitions];

            for (int p = 0; p < numPartitions; p++) {//loop over smaller groups of instances
                //gather this set of partitionSize instances
                Instances smallTestInstances = new Instances(testInstances, partitionSize);
                for (int inst = p * partitionSize; inst < partitionSize * (p + 1) && inst < testInstances.numInstances(); inst++)
                    smallTestInstances.add(testInstances.instance(inst));
                smallTestInstancesArray[p] = smallTestInstances;
            }
        }

        for (int i = 0; i < estimatorsForAttributes.length; i++) {
            // Get P(x) for entire array of values of attribute i
            double[] attrProbs;

            if (goingToSubdivide) {

                attrProbs = new double[testInstances.numInstances()];
                for (int p = 0; p < numPartitions; p++) {
                    double[] smallAttrVals = smallTestInstancesArray[p].attributeToDoubleArray(i);
                    double[] smallAttrProbs = estimatorsForAttributes[i].probabilityOf(smallAttrVals);
                    System.arraycopy(smallAttrProbs, 0, attrProbs, p * partitionSize, smallAttrProbs.length);
//                    System.err.print(".");
                }

            } else {
                double[] attrVals = testInstances.attributeToDoubleArray(i);
                attrProbs = estimatorsForAttributes[i].probabilityOf(attrVals);
            }
//            System.err.print("\n");

            // Check for NaNs
            for (int j = 0; j < attrProbs.length; j++) {
                if (Double.isNaN(attrProbs[j])) {
                    Instance inst = testInstances.instance(j);
                    System.err.println("Error (probably a missing value): got NaN for probability density of attr " + i + ", instance " + inst);
                }
            }

            if (useSmoothing) {
            	attrProbs = smoothDensityEstimates(attrProbs);
            }

            for (int j = 0; j < attrProbs.length; j++) {
                currentLogProbs[j] += Math.log(attrProbs[j]); // currentProb *= attrProb

                if (printInfinityDebugInfo && Double.isInfinite(Math.log(attrProbs[j])))
                    System.err.println("Infinity at item " + j + ", attribute " + (i + 1) +
                            ", testInstance is " + testInstances.instance(j));
            }
        }

//        for (Estimator estimator : estimatorsForAttributes) {
//            if (estimator instanceof KDEstimator) {
//                File f = new File(((KDEstimator) estimator).fileName);
//                f.delete();
//            }
//        }

        return currentLogProbs;
    }


    protected double[] smoothDensityEstimates(double[] original) {
        double[] smoothed = new double[original.length];
        double minOffset = 0.0001;    // note: these scores aren't constrained to [0, 1] anyway, so initializing minOffset = 1
                                      // isn't particularly meaningful. Might as well start out with a small value.
        for (int i = 0; i < smoothed.length; i++) {
            if (original[i] < minOffset && original[i] > 0)
                minOffset = original[i];
        }
        if (minOffset / 2 > 0)
            newValToUseSmoothing = minOffset / 2;
        else if (minOffset * .8 > 0)
            newValToUseSmoothing = minOffset * .8;
        else newValToUseSmoothing = minOffset;

        for (int i = 0; i < original.length; i++) {
            smoothed[i] = original[i];
            if (smoothed[i] < minOffset) {    // P(x | A) is almost 0
                if (printInfinityDebugInfo) {
                    System.err.println("Smoothing item " + i + ": orig P(x | A) = " + Double.toString(smoothed[i]) +
                            "; new is " + Double.toString(newValToUseSmoothing));
                }
                smoothed[i] = newValToUseSmoothing;
            }
        }

        return smoothed;
    }
    
    public DataGenerator getDataGenerator(){
    	return dataGen;
    }
}
