package cade;

import cade.classifiers.MultipleClassifierFDC;
import cade.experiments.SyntheticDataExperiments;
import cade.experiments.UCIExperiments;
import cade.generativeProbDensities.GenerativeProbDensity;
import cade.positiveDataGenerators.NoLabelsArffDataGenerator;
import weka.classifiers.evaluation.NominalPrediction;
import weka.core.Attribute;
import weka.core.Instances;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;

public class PseudoAnomalyGo {

    // todo: Move these both to Params(?)
    public static boolean printProgress = true;
    public static boolean unifBlanket = false;

    public static void main(String[] args) throws Exception {
        long startTime = System.currentTimeMillis();

        UCIExperiments.runUCIDataCrossValFromCommandLine(args);
//        runUnlabeledWithCommandLineArgs(args);  // for arbitrary data sets

        if (false) { // Other functions that can be called from the driver. You probably want one of these:
            // Our experiments with labeled data (supervised or unsupervised); computes AUCs.
            UCIExperiments.runUCIDataCrossValFromCommandLine(args);
            UCIExperiments.runUCIDataCrossValFromCommandLine(new String[]{"1", "0", "true", "."});
            UCIExperiments.runUCIDataUnsupFromCommandLine(args);//new String[]{"1", "0", "."}

            // Fully unlabeled data; simply prints predictions.
            runUnlabeledWithCommandLineArgs(args);
            runUnlabeledWithCommandLineArgs(new String[]{"/Users/agentzel/Documents/workspace/pseudo-anomaly/oct-up.arff", "/Users/agentzel/Desktop/octUpPreds", "0", "3", "-1", "1,2"});//new String[]{"../data/UCIDataSets/pendigits.arff", "./tempyPredictions.csv", "0", "3","-1", "1"}

            // Our experiments with synthetic data
            SyntheticDataExperiments.runGaussianMixtureExperiments(args);
            SyntheticDataExperiments.runVaryingDimensionsExperiments(args);
        }

        Parameters.rCaller.stopRCallerOnline(); // once and for all
        System.out.println("Total runtime in milliseconds: " + (System.currentTimeMillis() - startTime));
    }

    // This method is appropriate for when the data file does not provide a labeled ground truth.
    // It does not compute AUC, only predictions.
    private static void runUnlabeledWithCommandLineArgs(String[] args) throws Exception {
        String inputFormat = "Command-line arguments:\n" +
                "java -cp [weka and RCaller] PseudoAnomalyGo\n" +
                "    inputFile.[csv|arff] outputPredictions.csv classifierType pseudoAnomalyType numTrainingInstances IDattrs [attrsToUse] \n\n" +
                "where: classifierType is integer 0-5 (see list Parameters.ClassifierType; 0 (RF) & 1 (KNN) tend to perform well),\n" +
                "       pseudoAnomalyType is integer 0-4 (see list Parameters.PseudoNegGenerationType; 0 (UNIF - fast), 1 & 2 (indep KDEs & Bayes Net - slow) tend to perform well),\n" +
                "       numTrainingInstances is number we'll sample from inputFile to build model (-1 to use all)\n" +
                "       IDattrs is comma-separated list of column(s) we'll print out as identifiers (use -1 for none; if there's a class attr, include it here to prevent it from being used),\n" +
                "       attrsToUse is optional comma-separated list of columns to use in classification (default or -1: all except IDattrs)";

        if (args.length < 6) {
            System.out.println("not enough arguments to command line");
        }

        String inputFile, outputFile;
        int classifierCode, pNegCode, numTrainingInstances;
        int[] idAttributes, attrsToUse;

        try {   // Parse command-line args
            inputFile = args[0];
            outputFile = args[1];
            classifierCode = Integer.parseInt(args[2]);
            pNegCode = Integer.parseInt(args[3]);

            numTrainingInstances = Integer.parseInt(args[4]);
            String IDAttrsString = args[5];
            idAttributes = DriverUtils.convertStringToIntArray(IDAttrsString);
            if (idAttributes.length < 1) {
                throw new Exception("Need to specify at least one column that contains IDs"); // todo: really?
            }

            String attrsToUseString = "";
            if (args.length >= 7) {
                attrsToUseString = args[6].equals("-1") ? "" : args[6];
            }
            attrsToUse = DriverUtils.convertStringToIntArray(attrsToUseString);

            if (args.length > 7) {
                System.out.println("Ignoring some arguments; expected 6 or 7, but found " + args.length);
            }

        } catch (Exception ex) {
            ex.printStackTrace();
            System.out.println(inputFormat);
            return;
        }

        // Print the configs and set up config variables
        System.out.println("Run started at " + new java.util.Date());
        System.out.println("Using columns " + DriverUtils.convertBackToString(idAttributes) + " as identifiers");
        if (attrsToUse.length > 0) {
            System.out.println("Using columns " + DriverUtils.convertBackToString(attrsToUse) + " as inputs to model");
        } else {
            System.out.println("Using all other columns as inputs to model");
        }

        //if printing output to file, open writer immediately to verify that file path works
        BufferedWriter writer = null;
        if (outputFile != null) {
            writer = new BufferedWriter(new FileWriter(outputFile));
            System.out.println("Predictions will be written to file " + outputFile);
        }

        ParamsMultiRuns paramsMeta = new ParamsMultiRuns(classifierCode, pNegCode);
        int settingParams = 19;
        Parameters params = new Parameters(inputFile, attrsToUse, idAttributes, numTrainingInstances, settingParams, paramsMeta);

        int i = 0;  // no looping in this version
        System.out.println("classifierType: " + paramsMeta.classifierTypesThisRun[i]);
        params.setPseudoNegTypeFromIndex(i);
        System.out.print("pseudoNegGenerationType: " + params.pseudoNegType);
        if (paramsMeta.pseudoNegGenerationTypesThisRun[i] == Parameters.PseudoNegGenerationType.UNIFORM)
            System.out.print(" (Range: " + params.uniformPNegRange + ")");
        System.out.println();


        // --> Actually run!
        runClassifierAndPrintPredictions(params, writer);

    }

    // Simple no loops version (similar to function used in test cases).
    // Expects to be called with the 'NoLabelsArffDataGenerator' style (no true class labels) -- in which the test set
    // is the entire data set. [might conceivably be made to work with labeled data, using numFolds = 1, but the
    // two classes of data generators work differently...]
    // Expects to be called with a single classifier. [might conceivably be made to work with several.]
    public static void runClassifierAndPrintPredictions(Parameters params, BufferedWriter writer) throws Exception {
        NoLabelsArffDataGenerator trueDataGen = (NoLabelsArffDataGenerator) params.getDataGenerator();
        Instances posTrainInstances = trueDataGen.generateTrainingPositives(-1);

        // Give them to classifiers
        MultipleClassifierFDC classifier = params.paramsMeta.getClassifier();

        classifier.addTrainingData(posTrainInstances);

        // Generate pseudo-negatives
        GenerativeProbDensity pseudoNegGen = params.getPseudoNegativeGenerator();
        Instances pseudoNegInstances = pseudoNegGen.generateItems(-1);

        // Give them to classifiers
        classifier.addTrainingData(pseudoNegInstances);

        // Build classifiers
        System.out.println("building classifier...");
        classifier.buildClassifier();

        Instances testInstances = trueDataGen.generateTestingData(-1);
        System.out.println("getting predictions from classifier for " + testInstances.numInstances() + " instances...");

        classifier.constructFormulaDensityCombiners(params.getProbDensity(), params.countZeroes, params.smoothToRemove1sAnd0s);
        ArrayList<NominalPrediction>[] predictions = classifier.getPredictionsForTestSet(testInstances);

        //now print predictions to file, with ids first
        String[][] idAttributeValues = trueDataGen.idAttrValues;    // previously stored. rows are still in the same order

        System.out.println("writing predictions...");
        for (int inst = 0; inst < predictions[0].size(); inst++) {
            for (int id = 0; id < idAttributeValues.length; id++) {
                if (id != 0)
                    writer.write(",");
                writer.write(idAttributeValues[id][inst]);
            }
            // This is ranked with highest = most normal
//        	writer.write("," + predictions[0].get(inst).distribution()[0] + "\n");
// todo: clarify/document what scores are actually being printed out
            // Print out the P(anomalous). To get that, let b = distribution()[0], which
            // is log(P(positive)). Compute 1 - exp(b).
            // (We could use distribution()[1] directly as a ranking, but can't directly convert it to a probability.
            double probAnomalous = 1 - Math.exp(predictions[0].get(inst).distribution()[0]);
            // ack. that didn't work, because when we subtracted a very small number from 1, we lost precision and it came out to 1 again.
            // Well, use distribution()[1] so it's in the right order, and then give the best we can do (i.e., rounded off) for a probability.
            writer.write("," + predictions[0].get(inst).distribution()[1] + "," + probAnomalous + "\n");
        }
        writer.flush();
        writer.close();
    }

    /** todo: ugh, no, putting the columns into new Instances objects won't work. the savedIDAttrs only contain header info.
     * Prints (to file) the individual predictions for each instance.
     * Order of the columns: idAttributeValues (if present), testInstances
     * @param testInstances: instances we've computed predictions for. If they contain a class label, it'll be printed too.
     * @param predictions: array of 1 or more sets of predictions. Each set of predictions will become 1 column in output.
     *                   It's expected to be the same length and in the same order as testInstances.
     * @param outputFile: if non-null, print to file; otherwise, use stdout.
     * @param savedIDAttrs: if non-null, print these as first columns.
     */
    protected void saveInstancesWithPredictions(Instances testInstances, ArrayList<NominalPrediction>[] predictions,
                                                File outputFile, Attribute[] savedIDAttrs) {
        if (savedIDAttrs != null) {
            int numInserted = 0;
            for (Attribute attr : savedIDAttrs) {
                testInstances.insertAttributeAt(attr, numInserted);
                numInserted++;
            }
        }
        // todo: convert predictions to attribute(s) and insert them
    }

}
