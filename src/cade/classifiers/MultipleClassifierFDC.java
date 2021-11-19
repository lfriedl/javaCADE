package cade.classifiers;

import java.util.ArrayList;


import cade.Evaluator;
import cade.FormulaDensityCombiner;
import cade.Parameters;
import cade.StatsZeroCounter;
import cade.generativeProbDensities.GenerativeProbDensity;
import weka.classifiers.evaluation.NominalPrediction;
import weka.core.Instances;

// This class encapsulates (i.e., prevents the driver from needing to think about) the loop over classifiers.
// It holds multiple LocalClassifiers and multiple FormulaDensityCombiners.
// (Despite similarities to the API, it's no longer like a LocalClassifier; it's more like a FormulaDensityCombiner.)
public class MultipleClassifierFDC {
    public int numClassifiers;
    public LocalClassifier[] classifiersToCall;
    Parameters.ClassifierType[] classifierTypes;

    public FormulaDensityCombiner[] fdcs;
    StatsZeroCounter[] statsZeroCounters;

    public MultipleClassifierFDC(Parameters.ClassifierType[] classifierTypes, StatsZeroCounter[] statsZeroCounters) {
        numClassifiers = classifierTypes.length;
        classifiersToCall = new LocalClassifier[numClassifiers];
        this.classifierTypes = classifierTypes;
        this.statsZeroCounters = statsZeroCounters;
        for (int i = 0; i < numClassifiers; i++) {
            if (classifierTypes[i] == Parameters.ClassifierType.NOCLASSIFIER)
                classifiersToCall[i] = new NoClassifier(null);
            else
                classifiersToCall[i] = new LocalClassifier(classifierTypes[i]);
        }
    }

    public boolean addTrainingData(Instances newData) {
         for (LocalClassifier classifier : classifiersToCall) {
             if (!classifier.addTrainingData(newData)) {
                 return false;
             }
         }
         return true;
     }

    public void buildClassifier() throws Exception {
        for (LocalClassifier classifier : classifiersToCall) {
            classifier.buildClassifier();
       }
    }

    public void constructFormulaDensityCombiners(GenerativeProbDensity pseudoNegativeGenerator,
                                                 boolean countZeroes, boolean useSmoothing) {
        fdcs = new FormulaDensityCombiner[numClassifiers];
        for (int i = 0; i < numClassifiers; i++)
            fdcs[i] = new FormulaDensityCombiner(classifiersToCall[i], pseudoNegativeGenerator, countZeroes, useSmoothing);
    }

    public ArrayList<NominalPrediction>[] getPredictionsForTestSet(Instances testInstances) {
    	@SuppressWarnings("unchecked")
		ArrayList<NominalPrediction>[] predictionLists = new ArrayList[numClassifiers];
    	for (int i = 0; i < numClassifiers; i++) {
    		predictionLists[i] = fdcs[i].getPredictionsArrayForTestSet(testInstances, statsZeroCounters[i]);
    	}
    	return predictionLists;
    }

	public double[] getAUCFromPredArray(ArrayList<NominalPrediction>[] preds) {
		double[] aucs = new double[numClassifiers];
    	for (int i = 0; i < numClassifiers; i++) {
    		try {
//				aucs[i] = (new Evaluator()).getAUCFromPredArrayUsingPerl(preds[i]);
				aucs[i] = Evaluator.getAUCFromWeka(preds[i]);
			} catch (Exception e) {
				e.printStackTrace();
			}
    	}
    	return aucs;
	}

    // Used just rarely, for when there's no FDC
    public double[] getAUCForTestSetClassifierAlone(Instances testInstances) throws Exception {
        double[] aucs = new double[numClassifiers];
        for (int i = 0; i < numClassifiers; i++) {
            LocalClassifier classifier = classifiersToCall[i];
            aucs[i] = (classifier.getAUCForTestSet(testInstances))[0];
        }
        return aucs;
    }

	public int howManyClassifiersAreYou() { return numClassifiers; }
}
