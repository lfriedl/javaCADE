package cade;

import cade.classifiers.LocalClassifier;
import cade.generativeProbDensities.GenerativeProbDensity;
import cade.positiveDataGenerators.DataGenerator;
import weka.classifiers.evaluation.NominalPrediction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Random;


/**
 * Key equation from paper:
 *
 * T = training class
 * A = pseudo negatives (i.e., "our estimate" of T)
 * want: P(X | T)
 * P(X | T ) = (1 - P(T)) P(T | X) P(X | A)
 *              ----- [over] ----
 *              P(T) (1 - P(T | X))
 *
 *
 * K. Hempstalk, E. Frank, I. H. Witten.
 * One-Class Classification by Combining Density and Class Probability Estimation.
 * ECML PKDD 2008.
 */
public class FormulaDensityCombiner {
	Random rng;
	
    boolean useSmoothing = true;

    // For experiments / occasional
    boolean printInstanceScoreComponents = false;
    boolean printInfinityDebugInfo = false;
    
    protected GenerativeProbDensity probDensity;
    protected LocalClassifier classifier;
    
    //these are used when counting the number of zeroes
    public boolean printOutZeroCounts = false;

    protected double trainingPercentPositive = .5;


    public FormulaDensityCombiner(LocalClassifier classifier, GenerativeProbDensity pDens, boolean countZeroes, boolean useSmoothing){
    	probDensity = pDens;
    	this.classifier = classifier;
    	printOutZeroCounts = countZeroes;
    	this.useSmoothing = useSmoothing;
    }

    public FormulaDensityCombiner() {
    }

    public void setRandom(Random r) {
        rng = r;
    }
    
    public double[] computeLogDensity(Instances testInstances){
    	double[][] classifierPredictions = null;
    	try {
    		classifierPredictions = classifier.getDistributionsForTestSet(testInstances);
    	} catch (Exception e) {
    		System.out.println("EXCEPTION!!  Run awaaaay!");
    		e.printStackTrace();
    	}
    	
    	double[] currentLogProbs = probDensity.computeLogDensity(testInstances);
    
    	if (useSmoothing) {
//            System.out.println("Smoothing classifier predictions");
            classifierPredictions = smoothClassifierScores(classifierPredictions);
        }
    	
    	double[] finalDensities = new double[currentLogProbs.length];
    	
    	for (int i = 0; i < testInstances.numInstances(); i++) {
            Instance inst = testInstances.instance(i);
            NominalPrediction nomPred;
            nomPred = scoreForInstance(inst, classifierPredictions[i], currentLogProbs[i]);
            finalDensities[i] = nomPred.distribution()[0];

            if (printInfinityDebugInfo) {
                if (Double.isInfinite(currentLogProbs[i]))
                    System.err.println("Item " + i + " had a +/-Infinity from the density estimate; Instance is " + inst);
                else if (Double.isInfinite(nomPred.distribution()[0]))
                    System.err.println("Item " + i + " had a +/-Infinity prediction (probably from the classifier); Instance is " + inst);
            }
        }

        return finalDensities;
    }

     // Methods to do the same things as before, but with access to all the test data at once.
    public ArrayList<NominalPrediction> getPredictionsArrayForTestSet(Instances testInstances, StatsZeroCounter stats) {
    	double[][] testSetPredictions = null;
		try {
			testSetPredictions = classifier.getDistributionsForTestSet(testInstances);
		} catch (Exception e) {
			System.out.println("EXCEPTION!!!! :-O");
			e.printStackTrace();
		}

        ArrayList<NominalPrediction> predArray = new ArrayList<NominalPrediction>();
        double[] currentLogProbs = probDensity.computeLogDensity(testInstances);

        if (useSmoothing) {
//            System.out.println("Smoothing classifier predictions");
            testSetPredictions = smoothClassifierScores(testSetPredictions);
        }

        if (printOutZeroCounts) {
            // counting is done after all smoothing (b/c it wasn't possible to do it before smoothing density estimate,
            // which in turn is because we needed to smooth each attr's density estimate before taking its log prob)
            stats.countZeroes(testInstances, testSetPredictions, currentLogProbs);
        }

        for (int i = 0; i < testInstances.numInstances(); i++) {
            Instance inst = testInstances.instance(i);
            NominalPrediction nomPred;
            nomPred = scoreForInstance(inst, testSetPredictions[i], currentLogProbs[i]);
            predArray.add(nomPred);

            if (printInfinityDebugInfo) {
                if (Double.isInfinite(currentLogProbs[i]))
                    System.err.println("Item " + i + " had a +/-Infinity from the density estimate; Instance is " + inst);
                else if (Double.isInfinite(nomPred.distribution()[0]))
                    System.err.println("Item " + i + " had a +/-Infinity prediction (probably from the classifier); Instance is " + inst);
            }
        }

         if (printOutZeroCounts) {
            stats.countTieScores(testSetPredictions, predArray);
        }

        return predArray;
    }

    // logit = log(x / (1-x))
    // A neat way to do this even when x or (1 - x) are very small: stick with computations near 0, not near 1.
    // (Variables too close to 1 may round off to 1; but we can do valid operations on values that are equally close to 0.)
    private double logit(double x, double one_minusX) {
        if (x > .25 && one_minusX > .25) {
            // old fashioned
            return Math.log(x / one_minusX);
        } else if (x < .25) {    // log(x) - log(1-x)
            return Math.log(x) - Math.log1p(-1 * x);
        } else {                 // x > .75, but (1 - x) < .25
            // Can call log(1-x), but want to replace log(x) with log1p(-(1-x))
            // Note: the sample code, http://www.codeproject.com/script/Articles/ViewDownloads.aspx?aid=25294,
            // didn't specially handle this case. But we might have a decent value for one_minusX and not for x.
            return Math.log1p(-1 * one_minusX) - Math.log(one_minusX);
        }

    }


    private NominalPrediction scoreForInstance(Instance inst, double[] classifierPredForInstance, double pNegDensityFromModel) {

        // Try to avoid any infinities.
        double p_t_x = classifierPredForInstance[0];
        // Sometimes classifierPredForInstance[0] == 1, yet classifierPredForInstance[1] > 0 by a tiny amount.
//        double oneminus_p_t_x = 1 - p_t_x;
        double oneminus_p_t_x = classifierPredForInstance[1];
        double betterLogit_p_t_x = logit(p_t_x, oneminus_p_t_x);

        double log_p_x_a = pNegDensityFromModel;

        double p_t = trainingPercentPositive;
        double log_p_t = Math.log(p_t);
        double log1minus_p_t = Math.log(1 - p_t);

        if (printInfinityDebugInfo) {   // prints something when logit() saved us
            double log_p_t_x = Math.log(p_t_x);
            double log1minus_p_t_x = Math.log(oneminus_p_t_x);
            if (Double.isInfinite(log_p_t_x) || Double.isInfinite(log1minus_p_t_x))
                System.err.println("Would have had an Infinity here. Classifier's P(pos) is " + Double.toString(p_t_x) +
                        ", and P(neg) is " + Double.toString(oneminus_p_t_x) +
                        "; new logit gives " + Double.toString(betterLogit_p_t_x) + ", for instance " + inst);
        }


        // Use the equation from top of file
//        double p_x_t = ((1 - p_t) / p_t) * (p_t_x / (1 - p_t_x)) * p_x_a;
        //double log_p_x_t = (log1minus_p_t - log_p_t) + (log_p_t_x - log1minus_p_t_x) + log_p_x_a;
        double log_p_x_t = (log1minus_p_t - log_p_t) + betterLogit_p_t_x + log_p_x_a;

        // Catch NaN's here
        if (Double.isNaN(log_p_x_t) && !Double.isNaN(log_p_x_a)) {
            // There's only 1 way this can happen. P(X | A) = 0, but P(T | X) = 1. In the computation,
            // that turned into Infinity + -Infinity = NaN.
            // Correct thing to do here (I think): density estimate should win, and we
            // should say P(X | T) = 0
            log_p_x_t = Double.NEGATIVE_INFINITY;
        }

        // build a NominalPrediction saying:
        // 1st score = log(P(positive)), so higher when more positive. max = 0.
        // 2nd score: higher when more anomalous. I'll just flip the sign of the 1st score rather
        // than compute anything meaningful for it.
        NominalPrediction nomPred = new NominalPrediction(inst.classValue(),
                new double[]{log_p_x_t, -1 * log_p_x_t});

        return nomPred;
    }

    
    // Smoothing routines

    // Used with testSetPredictions, where each instance has an array, with original[i][0] = P(0) and original[i][1] = P(1)
    // Now using the data to find the score closest to zero that we'll set the current 0's just below.
    // Doing things symmetrically -- i.e., even if one value in the distribution rounds off to 1, the other still holds info,
    // so store it wherever it works.
    private double[][] smoothClassifierScores(double[][] original) {
        double[][] smoothed = new double[original.length][original[0].length];
        double minOffset = 1;
        for (int i = 0; i < original.length; i++) {
            if (original[i][0] < minOffset && original[i][0] > 0)
                minOffset = original[i][0];
            else if (original[i][1] < minOffset && original[i][1] > 0)
                minOffset = original[i][1];
        }
        double newValToUse;
        if (minOffset / 2 > 0)
            newValToUse = minOffset / 2;
        else if (minOffset * .8 > 0)    // these "else"s are probably redundant, but they haven't been tested
            newValToUse = minOffset * .8;
        else newValToUse = minOffset;

        for (int i = 0; i < original.length; i++) {
            smoothed[i] = original[i];
            if (smoothed[i][0] < minOffset) {    // P(+) is almost 0
                if (printInfinityDebugInfo) {
                    System.err.println("Smoothing item " + i + ": orig P(+) = " + Double.toString(smoothed[i][0]) +
                            "; P(-) = " + Double.toString(smoothed[i][1]) + "; new P(+) = " + Double.toString(newValToUse));
                }
                smoothed[i][0] = newValToUse;
                smoothed[i][1] = 1 - newValToUse;

            } else if (smoothed[i][1] < minOffset) {
                if (printInfinityDebugInfo) {
                    System.err.println("Smoothing item " + i + ": orig P(+) = " + Double.toString(smoothed[i][0]) +
                            "; P(-) = " + Double.toString(smoothed[i][1]) + "; new P(-) = " + Double.toString(newValToUse));
                }
                smoothed[i][1] = newValToUse;
                smoothed[i][0] = 1 - newValToUse;
            }
        }

        return smoothed;
    }

    
    public DataGenerator getDataGenerator(){
    	return probDensity.getDataGenerator();
    }

	public String getName(){
		return probDensity.getName() + "/" + classifier.toString();
	}

}


