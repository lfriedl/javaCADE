package cade;

import cade.classifiers.MultipleClassifierFDC;
import cade.experiments.SyntheticDataExperiments;
import cade.experiments.UCIExperiments;
import cade.generativeProbDensities.GenerativeProbDensity;
import cade.positiveDataGenerators.NoLabelsArffDataGenerator;
import weka.classifiers.evaluation.NominalPrediction;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.ArrayList;
import java.util.Arrays;

public class PseudoAnomalyGo {

    // todo: Move these both to Params(?)
    public static boolean printProgress = true;
    public static boolean unifBlanket = false;

    public static void main(String[] args) throws Exception {
        long startTime = System.currentTimeMillis();

        runUnlabeledWithCommandLineArgs(args);  // for arbitrary data sets

        if (false) { // Other functions that can be called from the driver. You probably want one of these:
            // Our experiments with labeled data (supervised or unsupervised); computes AUCs.
            UCIExperiments.runSupervisedFromCommandLine(args);
            UCIExperiments.runSupervisedFromCommandLine(new String[]{"1", "0", "true", "."});
            UCIExperiments.runLabeledUnsupFromCommandLine(args);//new String[]{"1", "0", "."}

            // Fully unlabeled data; simply prints predictions.
            runUnlabeledWithCommandLineArgs(args);

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

        // Parse args, set up config variables
        Parameters params = parseCmdLineArgsUnlabeled(args);

        // open output file immediately to verify that the path works
        String outputFile = args[1];
        ConverterUtils.DataSink dataSink = new ConverterUtils.DataSink(outputFile);
        System.out.println("Predictions will be written to file " + outputFile);

        // Actually run!
        runClassifierAndPrintPredictions(params, dataSink);

    }

    private static Parameters parseCmdLineArgsUnlabeled(String[] args) throws Exception {
        String inputFormat = "Command-line arguments:\n" +
                "java [-classpath weka and RCaller] PseudoAnomalyGo inputFile.[csv|arff] outputPredictions.[csv|arff] classifierType pseudoAnomalyType numTrainingInstances IDattrs [attrsToUse] \n\n" +
                "e.g., java PseudoAnomalyGo inputFile.csv outputPredictions.csv 0 0 -1 -1\n\n" +
                "where:\n" +
                "       input and output files are in formats Weka can recognize (e.g., csv or arff)\n" +
                "       classifierType is a 0-based index into this list: "  + Arrays.toString(Parameters.ClassifierType.values()) + "\n" +
                "           classifierTypes 0 (RF) and 1 (KNN) tend to perform well\n" +
                "       pseudoAnomalyType is a 0-based index into this list: " + Arrays.toString(Parameters.PseudoNegGenerationType.values()) + "\n" +
                "           pseudoAnomalyTypes 0 (UNIF - fast), 1 (indep KDEs) and 2 (Bayes Net - slow) tend to perform well\n" +
                "       numTrainingInstances is number we'll sample from inputFile to build model (-1 to use all)\n" +
                "           (note: regardless of numTrainingInstances, all instances will be scanned to get data ranges)\n" +
                "       IDattrs is a comma-separated list of column(s) for the model to ignore (e.g., row identifiers)\n" +
                "           1-based index. Use -1 for none. These will be printed in the output file.\n" +
                "       attrsToUse (optional) is a comma-separated list of columns for the model to use\n" +
                "           1-based index. Default or -1: use all except IDattrs.\n\n" +
                "       After running, outputPredictions will contain a new attribute 'CADE_loglik'. A lower score means more anomalous.\n";



        String inputFile;
        int classifierCode, pNegCode, numTrainingInstances;
        int[] idAttributes, attrsToUse;

        try {   // Parse command-line args

            if (args.length < 6) {
                throw new Exception("not enough arguments to command line");
            }

            inputFile = args[0];
//            outputFile = args[1];
            classifierCode = Integer.parseInt(args[2]);
            pNegCode = Integer.parseInt(args[3]);

            numTrainingInstances = Integer.parseInt(args[4]);
            String IDAttrsString = args[5];
            idAttributes = DriverUtils.convertStringToIntArray(IDAttrsString);
            if (idAttributes.length < 1) {
                throw new Exception("Need to specify column(s) of ID attributes (or -1 if none)");
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
            System.out.println(inputFormat);
            throw(ex);
        }

        System.out.println("Run started at " + new java.util.Date());
        System.out.println("Using columns " + DriverUtils.convertBackToString(idAttributes) + " as identifiers");
        if (attrsToUse.length > 0) {
            System.out.println("Using columns " + DriverUtils.convertBackToString(attrsToUse) + " as inputs to model");
        } else {
            System.out.println("Using all other columns as inputs to model");
        }


        ParamsMultiRuns paramsMeta = new ParamsMultiRuns(classifierCode, pNegCode);
        int settingParams = 19;  // usual defaults: smoothing, equal number of pNegs as positive training instances
        Parameters params = new Parameters(inputFile, attrsToUse, idAttributes, numTrainingInstances, settingParams, paramsMeta);

        // print config info
        int i = 0;  // no looping in this version
        System.out.println("classifierType: " + paramsMeta.classifierTypesThisRun[i]);
        params.setPseudoNegTypeFromIndex(i);
        System.out.print("pseudoNegGenerationType: " + params.pseudoNegType);
        if (paramsMeta.pseudoNegGenerationTypesThisRun[i] == Parameters.PseudoNegGenerationType.UNIFORM)
            System.out.print(" (Range: " + params.uniformPNegRange + ")");
        System.out.println();

        return params;
    }

    // Simple no loops version (similar to function used in test cases) -- expects a single classifier and pseudoNegGen method.
    // Expects to be called with the 'NoLabelsArffDataGenerator' style (no true class labels) -- in which the test set
    // is the entire data set -- and with a single classifier.
    public static void runClassifierAndPrintPredictions(Parameters params, ConverterUtils.DataSink dataSink) throws Exception {
        NoLabelsArffDataGenerator trueDataGen = (NoLabelsArffDataGenerator) params.getDataGenerator();
        Instances posTrainInstances = trueDataGen.generateTrainingPositives(params.numTrainPositives);

        // Give them to classifier(s)
        MultipleClassifierFDC classifier = params.paramsMeta.getClassifier();

        classifier.addTrainingData(posTrainInstances);

        // Generate pseudo-negatives
        GenerativeProbDensity pseudoNegGen = params.getPseudoNegativeGenerator();
        Instances pseudoNegInstances = pseudoNegGen.generateItems(params.numPseudoNegs);

        // Give them to classifiers
        classifier.addTrainingData(pseudoNegInstances);

        // Build classifiers
        System.out.println("building classifier...");
        classifier.buildClassifier();

        Instances testInstances = trueDataGen.generateTestingData(-1);
        System.out.println("getting predictions from classifier for " + testInstances.numInstances() + " instances...");

        classifier.constructFormulaDensityCombiners(params.getProbDensity(), params.countZeroes, params.smoothToRemove1sAnd0s);
        ArrayList<NominalPrediction>[] predictions = classifier.getPredictionsForTestSet(testInstances);

        saveInstancesWithPredictions(trueDataGen, testInstances, predictions[0], dataSink);

    }



    public static void saveInstancesWithPredictions(NoLabelsArffDataGenerator trueDataGen, Instances testInstances,
                                                    ArrayList<NominalPrediction> predictions, ConverterUtils.DataSink dataSink) throws Exception {
        System.out.println("merging predictions into input data and saving...");

        // delete the fake class label used during CADE
        if (testInstances.classIndex() >= 0) {
            int classIndexToRm = testInstances.classIndex();
            testInstances.setClassIndex(-1); // unset class index so that weka lets us remove this attr
            testInstances.deleteAttributeAt(classIndexToRm);
        }
        // put the ID attrs back in. (Rows are still in the same order.)
        Instances instsToPrint = Instances.mergeInstances(trueDataGen.savedIDAttrs, testInstances);


        // construct new attribute for the predictions, and fill it in
        Attribute predAttr = new Attribute("CADE_loglik", instsToPrint.numAttributes());
        instsToPrint.insertAttributeAt(predAttr, instsToPrint.numAttributes());
        for (int i = 0; i < instsToPrint.numInstances(); i++) {
            Instance inst = instsToPrint.instance(i);
            inst.setValue(predAttr, predictions.get(i).distribution()[0]);
        }

        dataSink.write(instsToPrint);

    }

}
