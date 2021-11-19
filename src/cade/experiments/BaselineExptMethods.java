package cade.experiments;

import cade.Parameters;
import cade.classifiers.MultipleClassifierFDC;
import cade.generativeProbDensities.GenerativeProbDensity;
import cade.generativeProbDensities.OneEverywhereProbDensity;
import cade.generativeProbDensities.UniformProbDensity;
import cade.lof.LOFWrapper;
import cade.positiveDataGenerators.DataGenerator;
import cade.positiveDataGenerators.GaussianMixtureDataGenerator;
import weka.classifiers.evaluation.NominalPrediction;
import weka.core.Instances;

import java.util.ArrayList;

public class BaselineExptMethods {

    // Code: identical to run() except training negatives are actually from the true negative generator
    public static double[] classifyPosFromTrueNeg(Parameters params, MultipleClassifierFDC classifier) throws Exception {
        // Get some positives + maybe some negatives from TrueDataGenerator as positive training data
        DataGenerator trueDataGen = params.getDataGenerator();
        Instances posTrainInstances = trueDataGen.generateTrainingPositives(params.numTrainPositives);

        // Give them to classifier
        classifier.addTrainingData(posTrainInstances);

        // Generate negatives
        Instances negInstances;
        negInstances = trueDataGen.generateTrainingNegatives(params.numPseudoNegs);

        // Give them to classifier
        classifier.addTrainingData(negInstances);

        // Build classifier
        classifier.buildClassifier();

        // Get test set of positives & negatives
        Instances posTestInstances = trueDataGen.generateTestingPositives(params.numTestPositives);
        Instances negTestInstances = trueDataGen.generateTestingNegatives(params.numTestNegatives);

        Instances allTestInstances = new Instances(posTestInstances);   // copies it
        // Add in the negatives
        for (int i = 0; i < negTestInstances.numInstances(); i++) {
            allTestInstances.add(negTestInstances.instance(i));
        }

        // to use MultipleClassifierFDC without pNegs, send in the "blank" pNegType.
        // (it seems unlikely that the smoothing method could affect AUC of classifier used alone.)
        classifier.constructFormulaDensityCombiners(new OneEverywhereProbDensity(params),
                params.countZeroes, params.smoothToRemove1sAnd0s);

        // Evaluate predictions (calculate AUC)
        double[] auc = classifier.getAUCForTestSetClassifierAlone(allTestInstances);
        return auc;

    }

    // Builds CADE density estimator as usual (using any combo of methods), then evaluates it by seeing how well
    // it can distinguish held-out positives from a "uniform blanket" of negatives.
    public static void runClassifierAgainstUnifBlanket(Parameters params, MultipleClassifierFDC classifier,
                                                       double[] aucs) throws Exception {
        System.out.println("running classifer vs. uniform blanket");
        DataGenerator trueDataGen = params.getDataGenerator();
        Instances posTrainInstances = trueDataGen.generateTrainingPositives(params.numTrainPositives);

        // Generate pos instances and give them to classifier
        classifier.addTrainingData(posTrainInstances);

        // Generate pseudo-negatives and give them to classifier
        GenerativeProbDensity pseudoNegGen = params.getPseudoNegativeGenerator();
        Instances pseudoNegInstances = pseudoNegGen.generateItems(params.numPseudoNegs);
        classifier.addTrainingData(pseudoNegInstances);

        // Build classifier
        classifier.buildClassifier();

        // Get test sets of positives & negatives
        Instances allPosBlanketTestInstances, posTestInstances = null, unifBlanketTestInstances = null;

//        if (trueDataGen instanceof dataGenerators.UnsupervisedArffDataGenerator)
//        	allPosBlanketTestInstances = ((dataGenerators.UnsupervisedArffDataGenerator) trueDataGen).generateTestingData(-1);
//        else {
        posTestInstances = trueDataGen.generateTestingPositives(params.numTestPositives);
        UniformProbDensity unifBlanket = new UniformProbDensity(params);
        // todo: how many to use?
        unifBlanketTestInstances = unifBlanket.generateItems(posTestInstances.numInstances());

        allPosBlanketTestInstances = new Instances(posTestInstances);   // copies it
        // Add in the negatives
        for (int i = 0; i < unifBlanketTestInstances.numInstances(); i++) {
            allPosBlanketTestInstances.add(unifBlanketTestInstances.instance(i));
        }
//        }

        classifier.constructFormulaDensityCombiners(params.getProbDensity(), params.countZeroes, params.smoothToRemove1sAnd0s);
        ArrayList<NominalPrediction>[] preds = classifier.getPredictionsForTestSet(allPosBlanketTestInstances);
        double[] auc = classifier.getAUCFromPredArray(preds);

        for (int i = 0; i < auc.length; i++) {
            aucs[i] = auc[i];
        }

    }


    //Training works as usual, but for testing, but there are no true negatives.
    //Instead, we get a ranking against a uniform blanket as our result.
    //Also, uses multiple pNeg methods, since we need to get predictions on the same test data
    //pass in an array of classifiers, one classifiers.MultipleClassifierFDC for each pNeg method
    static boolean actuallyRankAgainstPositives = true;

    public static ArrayList<NominalPrediction>[][] runClassifierAndRankAgainstUniform(Parameters params, MultipleClassifierFDC[] classifier, double[][] aucs, ArrayList<Double> truePredictions, LOFWrapper lofWrapper, ArrayList<NominalPrediction>[] lofPreds) throws Exception {

        DataGenerator trueDataGen = params.getDataGenerator();
        Instances posTrainInstances = trueDataGen.generateTrainingPositives(params.numTrainPositives);

        Instances posTestInstances = null, pNegTestInstances = null;
//        Instances allPosTrueNegTestInstances = null;
        // Get test sets of positives & negatives
        //We want the same test data for all pNeg methods, so we can generate this ahead of time to use later
        Instances trueNegTestInstances = null;
//        if (trueDataGen instanceof dataGenerators.UnsupervisedArffDataGenerator)
//        	allPosTrueNegTestInstances = ((dataGenerators.UnsupervisedArffDataGenerator) trueDataGen).generateTestingData(-1);
//        else {
        posTestInstances = trueDataGen.generateTestingPositives(params.numTestPositives);

        //we'll use a pseudo neg generator to get the test instances, since it's already set up to generate uniform instances
        GenerativeProbDensity uniformPNegGen = new UniformProbDensity(params);
        trueNegTestInstances = uniformPNegGen.generateItems(params.numTestNegatives);

//	        allPosTrueNegTestInstances = new Instances(posTestInstances);   // copies it
//	        // Add in the negatives
//		    for (int i = 0; i < trueNegTestInstances.numInstances(); i++) {
//		       	allPosTrueNegTestInstances.add(trueNegTestInstances.instance(i));
//		    }
//        }

        ArrayList<NominalPrediction>[][] allPreds = new ArrayList[params.paramsMeta.numPNegMethodsToUse][params.paramsMeta.numClassifiers];

        for (int pNeg = 0; pNeg < params.paramsMeta.numPNegMethodsToUse; pNeg++) {
            // Give positive training instances to classifiers
            classifier[pNeg].addTrainingData(posTrainInstances);

            // Generate pseudo-negatives
            params.setPseudoNegTypeFromIndex(pNeg);
            GenerativeProbDensity pseudoNegGen = params.getPseudoNegativeGenerator();
            Instances pseudoNegInstances = pseudoNegGen.generateItems(params.numPseudoNegs);

            // Give them to classifiers
            classifier[pNeg].addTrainingData(pseudoNegInstances);
            classifier[pNeg].buildClassifier();

            //get predictions
            classifier[pNeg].constructFormulaDensityCombiners(params.getProbDensity(), params.countZeroes, params.smoothToRemove1sAnd0s);

//			if (actuallyRankAgainstPositives){
            aucs[pNeg] = new double[]{-1, -1, -1, -1, -1, -1};
//			} else{
//	        		ArrayList<NominalPrediction>[] preds = classifier[pNeg].getPredictionsForTestSet(allPosTrueNegTestInstances);
//	        		aucs[pNeg] = classifier[pNeg].getAUCFromPredArray(preds);
//	      	}

            if (actuallyRankAgainstPositives)
                allPreds[pNeg] = classifier[pNeg].getPredictionsForTestSet(posTestInstances);
            else {
                //get predictions on uniform blanket
                allPreds[pNeg] = classifier[pNeg].getPredictionsForTestSet(trueNegTestInstances);
            }
        }

        ArrayList<Double> preds;
        if (actuallyRankAgainstPositives)
            preds = ((GaussianMixtureDataGenerator) trueDataGen).getTrueProbabilities(posTestInstances);
        else
            preds = ((GaussianMixtureDataGenerator) trueDataGen).getTrueProbabilities(trueNegTestInstances);
        for (Double pred : preds)
            truePredictions.add(pred);

        //LOF time!!
        if (lofWrapper != null) {
            lofWrapper.buildModels(posTrainInstances);
        }
        // Get test set predictions and AUC for LOF
        ArrayList<NominalPrediction>[] tempPreds;
        if (actuallyRankAgainstPositives)
            tempPreds = lofWrapper.getPredsForNewTestSet(posTestInstances);
        else
            tempPreds = lofWrapper.getPredsForNewTestSet(trueNegTestInstances);
        for (int j = 0; j < lofPreds.length; j++)
            for (int i = 0; i < tempPreds[j].size(); i++)
                lofPreds[j].add(tempPreds[j].get(i));

        return allPreds;
    }
}
