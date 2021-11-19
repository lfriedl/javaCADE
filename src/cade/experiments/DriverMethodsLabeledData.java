package cade.experiments;

import cade.DriverUtils;
import cade.Parameters;
import cade.ParamsMultiRuns;
import cade.PseudoAnomalyGo;
import cade.classifiers.MultipleClassifierFDC;
import cade.classifiers.NoClassifier;
import cade.generativeProbDensities.GenerativeProbDensity;
import cade.generativeProbDensities.OneEverywhereProbDensity;
import cade.lof.LOFWrapper;
import cade.positiveDataGenerators.DataGenerator;
import weka.classifiers.evaluation.NominalPrediction;
import weka.core.Instances;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;

public class DriverMethodsLabeledData {
    //Use all the instances in the data file (so just passes -1) (if classAttr = -1, it's the final attr)
    public static void runCrossValWithLabeledData(String dataFile, int classAttr, String[] positiveClasses,
                                                  String[] negativeClasses, int[] attributesToUse, boolean baseline,
                                                  String resultsFileForAvgs) throws Exception {
        runCrossValWithLabeledData(dataFile, classAttr, positiveClasses, negativeClasses, attributesToUse, -1, baseline, resultsFileForAvgs);
    }

    // Multiple runs, over combinations of [pseudoNegType][classifier][trialNum].
    // Params need to be set within Parameters and ParamsMultiRuns.
    // If baseline is true, runs classifyPosFromTrueNeg instead.
    public static void runCrossValWithLabeledData(String dataFile, int classAttr, String[] positiveClasses,
                                                  String[] negativeClasses, int[] attributesToUse, int numInstances,
                                                  boolean baseline, String resultsFileForAvgs) throws Exception {
        int settingParamsMeta = 17; // 10 for main experiments; 17 for playing around later
//        int settingParams = 20;  // 20 = post-refactoring; 21 for unsupervised
        int settingParams = 19;  // while playing around
        ParamsMultiRuns paramsMeta = new ParamsMultiRuns(settingParamsMeta);

        if (baseline) {
            System.out.println("Overall baseline");
            paramsMeta.numPNegMethodsToUse = 1;
            paramsMeta.baselineClassifierTypesThisRun = new Parameters.ClassifierType[]{};
        }

        //if printing output to file, open writer immediately to verify that file path works
        BufferedWriter writer = null;
        if (resultsFileForAvgs != null)
            writer = new BufferedWriter(new FileWriter(resultsFileForAvgs));


        double[][][] aucs = new double[paramsMeta.numPNegMethodsToUse][paramsMeta.numClassifiers][paramsMeta.numRunsPerSetting];

        Parameters params = new Parameters(dataFile, classAttr, positiveClasses, negativeClasses, attributesToUse, numInstances, settingParams, paramsMeta);

        if (PseudoAnomalyGo.printProgress) {
            System.out.println("Using paramsMeta setting = " + settingParamsMeta + " and params setting = " + settingParams + "; run started at " + new java.util.Date());
            System.out.println("Percent positive instances in training file = " + params.getDataGenerator().percentPositive);
            for (int i = 0; i < paramsMeta.numPNegMethodsToUse; i++) {
                System.out.print("pseudoNegGeneration" + i + ": " + paramsMeta.pseudoNegGenerationTypesThisRun[i]);
                if (paramsMeta.pseudoNegGenerationTypesThisRun[i] == Parameters.PseudoNegGenerationType.UNIFORM)
                    System.out.print(" (Range: " + params.uniformPNegRange + ")");
                System.out.println();
            }
            for (int i = 0; i < paramsMeta.classifierTypesThisRun.length; i++) {
                System.out.println("classifier" + i + ": " + paramsMeta.classifierTypesThisRun[i]);
            }
            for (int i = 0; i < paramsMeta.baselineClassifierTypesThisRun.length; i++)
                System.out.println("classifier" + (i + paramsMeta.classifierTypesThisRun.length) + ": " + paramsMeta.baselineClassifierTypesThisRun[i] + " training-data baseline");
        }

        // We will store and print out LOF results separately. Set up the results array here.
        LOFWrapper lofWrapper = paramsMeta.getLOFWrapper(0, params.getDataGenerator().getRandomGenerator());
        double[][] lofResults = new double[paramsMeta.numRunsPerSetting][0];
        if (lofWrapper != null)
            lofResults = new double[paramsMeta.numRunsPerSetting][lofWrapper.numMethodsIncluded];


        //loop over pseudo-negative types
        for (int pNegMethod = 0; pNegMethod < paramsMeta.numPNegMethodsToUse; pNegMethod++) {
            params.setPseudoNegTypeFromIndex(pNegMethod);

            // Will count things across all the folds
            paramsMeta.resetCountsOfZeroes();

            // multiple trials or folds
            for (int runNum = 0; runNum < paramsMeta.numRunsPerSetting; runNum++) {

                // within each fold, run the set of classifiers
                MultipleClassifierFDC classifier = paramsMeta.getClassifier();
                MultipleClassifierFDC baselineClassifier = paramsMeta.getBaselineClassifier();

                lofWrapper = paramsMeta.getLOFWrapper(pNegMethod, params.getDataGenerator().getRandomGenerator());
                double[] results = new double[paramsMeta.numClassifiers];

                if (baseline) {
                    classifier = paramsMeta.getClassifier();
                    results = BaselineExptMethods.classifyPosFromTrueNeg(params, classifier);
                } else if (PseudoAnomalyGo.unifBlanket)
                    BaselineExptMethods.runClassifierAgainstUnifBlanket(params, classifier, results);
                else
                    runClassifierAndPNegBaseline(params, classifier, baselineClassifier, results, lofWrapper, lofResults[runNum]);

                for (int classifierNum = 0; classifierNum < paramsMeta.numClassifiers; classifierNum++)
                    aucs[pNegMethod][classifierNum][runNum] = results[classifierNum];

                if (PseudoAnomalyGo.printProgress) {
                    for (int classifierNum = 0; classifierNum < paramsMeta.baselineClassifierTypesThisRun.length + paramsMeta.classifierTypesThisRun.length; classifierNum++)//for each classifier
                        System.out.println("pseudoNegGeneration" + pNegMethod + " trial" + runNum +
                                " classifier" + classifierNum + ": " + aucs[pNegMethod][classifierNum][runNum]);
                    if (lofWrapper != null)
                        for (int methNum = 0; methNum < lofResults[runNum].length; methNum++)
                            System.out.println("LOF trial" + runNum + " method" + methNum +
                                    ": " + lofResults[runNum][methNum]);
                }
                params.prepareDataGenForNextRun();
            }
            // After all the folds and all the classifiers (for 1 pNeg method):
            if (params.countZeroes)
                paramsMeta.printSummaryOfPredictions();
        }

        // compute averages over all trials per setting and print
        DriverUtils.printCADEResults(aucs, writer, paramsMeta);

        // avg & var for LOF trials in data structure: lofResults[runNum][lofMethod]
        if (lofResults[0].length > 0) {
            DriverUtils.printLOFResults(lofResults, writer);
        }

        if (writer != null) {
            writer.flush();
            writer.close();
        }

    }

    // Run both the regular classifier (train: pos vs. pseudo-neg, test: pos vs. neg), plus baselines (train: pos vs.
    // pseudo-neg, test: pos vs. pseudo-neg), plus LOF. Uses same exact data where applicable.
    // Either object, if empty, will quietly do nothing. If LOF is null, we don't run it.
    // Input arg aucs will be filled in by the method.
    public static void runClassifierAndPNegBaseline(Parameters params, MultipleClassifierFDC classifier,
                                                    MultipleClassifierFDC baselineClassifier, double[] aucs,
                                                    LOFWrapper lofWrapper, double[] lofResults) throws Exception {
        DataGenerator trueDataGen = params.getDataGenerator();
        Instances posTrainInstances = trueDataGen.generateTrainingPositives(params.numTrainPositives);

        // Give them to classifiers
        classifier.addTrainingData(posTrainInstances);
        baselineClassifier.addTrainingData(posTrainInstances);

        // Generate pseudo-negatives
        Instances pseudoNegInstances;
        GenerativeProbDensity pseudoNegGen = null;
        if (classifier.classifiersToCall.length == 1 && classifier.classifiersToCall[0] instanceof NoClassifier)
            pseudoNegInstances = null;
        else {
            pseudoNegGen = params.getPseudoNegativeGenerator();
            pseudoNegInstances = pseudoNegGen.generateItems(params.numPseudoNegs);
        }

        // Give them to classifiers
        if (pseudoNegInstances != null) {
            classifier.addTrainingData(pseudoNegInstances);
            baselineClassifier.addTrainingData(pseudoNegInstances);

            // Build classifiers
            classifier.buildClassifier();
            baselineClassifier.buildClassifier();
        }
        if (lofWrapper != null) {
            lofWrapper.buildModels(posTrainInstances);
        }


        // Get test sets of positives & negatives
        Instances posTestInstances = trueDataGen.generateTestingPositives(params.numTestPositives);
        Instances pNegTestInstances = new Instances(posTrainInstances, 0);
        if (baselineClassifier.numClassifiers != 0)
            pNegTestInstances = pseudoNegGen.generateItems(posTestInstances.numInstances());
        Instances trueNegTestInstances = trueDataGen.generateTestingNegatives(params.numTestNegatives);

        Instances allPosTrueNegTestInstances = new Instances(posTestInstances);   // copies it
        // Add in the negatives
        for (int i = 0; i < trueNegTestInstances.numInstances(); i++) {
            allPosTrueNegTestInstances.add(trueNegTestInstances.instance(i));
        }

        // for training the baseline
        Instances allPosPNegTestInstances = new Instances(posTestInstances);   // copies it
        // Add in the negatives
        for (int i = 0; i < pNegTestInstances.numInstances(); i++) {
            allPosPNegTestInstances.add(pNegTestInstances.instance(i));
        }

        // Skip this iteration if one of the test sets is non-existent
        if (posTestInstances.numInstances() == 0 || (trueNegTestInstances.numInstances() == 0 && classifier.numClassifiers != 0)) {
            Arrays.fill(aucs, -1);
            if (lofWrapper != null) {
                Arrays.fill(lofResults, -1);
            }
            return;
        }

        // Get test set predictions and AUC for classifiers
        classifier.constructFormulaDensityCombiners(params.getProbDensity(), params.countZeroes, params.smoothToRemove1sAnd0s);
        ArrayList<NominalPrediction>[] preds = classifier.getPredictionsForTestSet(allPosTrueNegTestInstances);
        double[] auc = classifier.getAUCFromPredArray(preds);

        // Get test set predictions and AUC for "training baseline" classifier
        //don't use Hempstalk here - we're just interested in how well the classifier does on the training data
        baselineClassifier.constructFormulaDensityCombiners(new OneEverywhereProbDensity(params), params.countZeroes, params.smoothToRemove1sAnd0s);
        ArrayList<NominalPrediction>[] baselinePreds = baselineClassifier.getPredictionsForTestSet(allPosPNegTestInstances);
        double[] aucBaseline = baselineClassifier.getAUCFromPredArray(baselinePreds);

        // Get test set predictions and AUC for LOF
        if (lofWrapper != null) {
            double[] lofResultsTmp = lofWrapper.getAUCsForNewTestSet(allPosTrueNegTestInstances);
            System.arraycopy(lofResultsTmp, 0, lofResults, 0, lofResults.length);
        }

        // Concatenate the return values
        System.arraycopy(auc, 0, aucs, 0, auc.length);
        System.arraycopy(aucBaseline, 0, aucs, auc.length, aucBaseline.length);

        if (pseudoNegGen != null) {
            pseudoNegGen.doneWithProbDensity(); // deletes any temp files it created
        }

    }
}
