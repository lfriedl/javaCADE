package cade.experiments;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;

import cade.Evaluator;
import cade.Parameters;
import cade.ParamsMultiRuns;
import cade.classifiers.MultipleClassifierFDC;
import cade.lof.LOFWrapper;
import cade.positiveDataGenerators.GaussianMixtureDataGenerator;
import weka.classifiers.evaluation.NominalPrediction;

/**
 * Old code moved out of the driver.
 */
public class SyntheticDataExperiments {

    //if you pass in args, arg[0] = directory to write output to, arg[1] = number of dimensions
    public static void runVaryingDimensionsExperiments(String[] args) throws Exception {
        int settingParamsMeta = 14;
        int settingParams = 17;
        int numDimensions = 5;
        ParamsMultiRuns paramsMeta = new ParamsMultiRuns(settingParamsMeta);
        System.out.println("Using paramsMeta setting = " + settingParamsMeta + " and params setting = " + settingParams + "; run started at " + new java.util.Date());

        // Store all results here so we can average them as needed at the end.
        // For easier access later, components are in this order:
        //   aucs[pseudoNegType][classifier][covarSetting][trialNum]

        for (int i = 0; i < paramsMeta.classifierTypesThisRun.length; i++)
            System.out.println("classifier" + i + ": " + paramsMeta.classifierTypesThisRun[i]);
        for (int i = 0; i < paramsMeta.baselineClassifierTypesThisRun.length; i++)
            System.out.println("classifier" + (i + paramsMeta.classifierTypesThisRun.length) + ": " + paramsMeta.baselineClassifierTypesThisRun[i] + " baseline");
        for (int i = 0; i < paramsMeta.numPNegMethodsToUse; i++) {
            System.out.print("pseudoNegGeneration" + i + ": " + paramsMeta.pseudoNegGenerationTypesThisRun[i]);
            if (paramsMeta.pseudoNegGenerationTypesThisRun[i] == Parameters.PseudoNegGenerationType.UNIFORM) {
                Parameters params = new Parameters(settingParams, paramsMeta, numDimensions, 0);
                System.out.print(" (Range: " + params.uniformPNegRange + ")");
            }
            System.out.println();
        }

        int numNonsenseDims = 0;
        String outFile = "";
        if (args.length > 2)
            numNonsenseDims = Integer.parseInt(args[2]);
        if (args.length > 1)
            numDimensions = Integer.parseInt(args[1]);
        if (args.length > 0)
            outFile = args[0];// + "/" + "syntheticDim" + numDimensions + ".txt";

        numNonsenseDims = 20;
        numDimensions = 5;


        BufferedWriter writer = new BufferedWriter(new FileWriter((args.length > 0) ? outFile : "../output/gaussianMixtureRankResults/varyingDimensions/marginalBN.txt"));
        writer.write("Num dimensions, ranking vs true for each classifier type, AUC for each classifier type");
        for (int itr = 0; itr < 20; itr++) {
            Parameters params = new Parameters(settingParams, paramsMeta, numDimensions, numNonsenseDims);

            MultipleClassifierFDC[] classifiers = new MultipleClassifierFDC[paramsMeta.numPNegMethodsToUse];
            //Need multiple copies of classifier, one for each pNeg method
            for (int pNeg = 0; pNeg < paramsMeta.numPNegMethodsToUse; pNeg++)
                classifiers[pNeg] = paramsMeta.getClassifier();

            //Let's do LOF too!
            // We will store and print out LOF results separately. Set up the results array here.
            LOFWrapper lofWrapper = paramsMeta.getLOFWrapper(0, params.getDataGenerator().getRandomGenerator());
            ArrayList<NominalPrediction>[] lofPreds = new ArrayList[2];
            lofPreds[0] = new ArrayList<NominalPrediction>();
            lofPreds[1] = new ArrayList<NominalPrediction>();


            double[][] aucs = new double[paramsMeta.numPNegMethodsToUse][paramsMeta.numClassifiers];
            ArrayList<Double> truePredictions = new ArrayList<Double>();
            //predictions has an array list of predictions for every pNeg/classifier combination
            ArrayList<NominalPrediction>[][] predictions = BaselineExptMethods.runClassifierAndRankAgainstUniform(params, classifiers, aucs, truePredictions, lofWrapper, lofPreds);

            //We have all the prediction lists we care about for this distribution of positives.
            //Now, convert to rankings and compare            
            double[][] trueRankingComparisons = new double[paramsMeta.numPNegMethodsToUse][paramsMeta.numClassifiers];
            for (int j = 0; j < paramsMeta.numPNegMethodsToUse; j++) {
                for (int i = 0; i < paramsMeta.numClassifiers; i++) {
                    if (paramsMeta.pseudoNegGenerationTypesThisRun[j] == Parameters.PseudoNegGenerationType.UNIFORM && paramsMeta.classifierTypesThisRun[i] == Parameters.ClassifierType.NOCLASSIFIER)
                        trueRankingComparisons[j][i] = -1;
                    else
//                        trueRankingComparisons[j][i] = Evaluator.computeAndCompareRankings2(predictions[paramsMeta.pNegTypeIndex(paramsMeta.pseudoNegGenerationTypesThisRun[j])][paramsMeta.classifierIndex(paramsMeta.classifierTypesThisRun[i])], truePredictions);
                        trueRankingComparisons[j][i] = Evaluator.computeAndCompareRankings2(predictions[j][i], truePredictions);
                }
            }

            double lofRankingComparisonReg = Evaluator.computeAndCompareRankings2(lofPreds[0], truePredictions);
            double lofRankingComparisonBagged = Evaluator.computeAndCompareRankings2(lofPreds[1], truePredictions);

            String toPrint = "" + numNonsenseDims;
            for (int i = 0; i < paramsMeta.numPNegMethodsToUse; i++)
                for (int j = 0; j < paramsMeta.numClassifiers; j++)
                    toPrint += " " + trueRankingComparisons[i][j];
            toPrint += " " + lofRankingComparisonReg + " " + lofRankingComparisonBagged;
            toPrint += "\n";


            System.out.print(toPrint);
            writer.write(toPrint);
            writer.flush();
//            writer.write(((cade.dataGenerators.GaussianMixtureDataGenerator)params.getDataGenerator()).printGaussians());
        }
        writer.flush();
    }

    // KEEP
    public static void runGaussianMixtureExperiments(String[] args) throws Exception {
        int settingParamsMeta = 14;
        int settingParams = 17;
        ParamsMultiRuns paramsMeta = new ParamsMultiRuns(settingParamsMeta);
        System.out.println("Using paramsMeta setting = " + settingParamsMeta + " and params setting = " + settingParams + "; run started at " + new java.util.Date());

        // Store all results here so we can average them as needed at the end.
        // For easier access later, components are in this order:
        //   aucs[pseudoNegType][classifier][covarSetting][trialNum]

        for (int i = 0; i < paramsMeta.classifierTypesThisRun.length; i++)
            System.out.println("classifier" + i + ": " + paramsMeta.classifierTypesThisRun[i]);
        for (int i = 0; i < paramsMeta.baselineClassifierTypesThisRun.length; i++)
            System.out.println("classifier" + (i + paramsMeta.classifierTypesThisRun.length) + ": " + paramsMeta.baselineClassifierTypesThisRun[i] + " baseline");
        for (int i = 0; i < paramsMeta.numPNegMethodsToUse; i++) {
            System.out.print("pseudoNegGeneration" + i + ": " + paramsMeta.pseudoNegGenerationTypesThisRun[i]);
            if (paramsMeta.pseudoNegGenerationTypesThisRun[i] == Parameters.PseudoNegGenerationType.UNIFORM) {
                Parameters params = new Parameters(settingParams, paramsMeta, 10, 0);
                System.out.print(" (Range: " + params.uniformPNegRange + ")");
            }
            System.out.println();
        }
        //./gaussianMixtureRankResults/gaussians3dim5EigenResults.txt

        String covMethod;
        if (args.length < 1)
            covMethod = "eigen";
        else
            covMethod = args[0];

        String outFileName;
        if (args.length < 2)
            outFileName = "./corrExperimentOutput.txt";
        else
            outFileName = args[1];


        BufferedWriter writer = new BufferedWriter(new FileWriter(outFileName));
        writer.write("Avg abs of correlation, ranking vs true for each pneg/classifier combo, bagged LOF ranking vs true");
        for (int itr = 0; itr < 100; itr++) {
            Parameters params = new Parameters(settingParams, paramsMeta, covMethod);

            MultipleClassifierFDC[] classifiers = new MultipleClassifierFDC[paramsMeta.numPNegMethodsToUse];
            //Need multiple copies of classifier, one for each pNeg method
            for (int pNeg = 0; pNeg < paramsMeta.numPNegMethodsToUse; pNeg++)
                classifiers[pNeg] = paramsMeta.getClassifier();

            //Let's do LOF too!
            // We will store and print out LOF results separately. Set up the results array here.
            LOFWrapper lofWrapper = paramsMeta.getLOFWrapper(0, params.getDataGenerator().getRandomGenerator());
            ArrayList<NominalPrediction>[] lofPreds = new ArrayList[2];
            lofPreds[0] = new ArrayList<NominalPrediction>();
            lofPreds[1] = new ArrayList<NominalPrediction>();

            double[][] aucs = new double[paramsMeta.numPNegMethodsToUse][paramsMeta.numClassifiers];
            ArrayList<Double> truePredictions = new ArrayList<Double>();
            //predictions has an array list of predictions for every pNeg/classifier combination
            ArrayList<NominalPrediction>[][] predictions = BaselineExptMethods.runClassifierAndRankAgainstUniform(params, classifiers, aucs, truePredictions, lofWrapper, lofPreds);

            //We have all the prediction lists we care about for this distribution of positives.
            //Now, convert to rankings and compare

            double[][] trueRankingComparisons = new double[paramsMeta.numPNegMethodsToUse][paramsMeta.numClassifiers];
            for (int j = 0; j < paramsMeta.numPNegMethodsToUse; j++) {
                for (int i = 0; i < paramsMeta.numClassifiers; i++) {
                    if (paramsMeta.pseudoNegGenerationTypesThisRun[j] == Parameters.PseudoNegGenerationType.UNIFORM && paramsMeta.classifierTypesThisRun[i] == Parameters.ClassifierType.NOCLASSIFIER)
                        trueRankingComparisons[j][i] = -1;
                    else
//                        trueRankingComparisons[j][i] = Evaluator.computeAndCompareRankings2(predictions[paramsMeta.pNegTypeIndex(paramsMeta.pseudoNegGenerationTypesThisRun[j])][paramsMeta.classifierIndex(paramsMeta.classifierTypesThisRun[i])], truePredictions);
                        trueRankingComparisons[j][i] = Evaluator.computeAndCompareRankings2(predictions[j][i], truePredictions);
                }
            }

            double lofRankingComparison = Evaluator.computeAndCompareRankings2(lofPreds[0], truePredictions);
            double lofRankingComparison2 = Evaluator.computeAndCompareRankings2(lofPreds[1], truePredictions);

            String toPrint = "" + ((GaussianMixtureDataGenerator) params.getDataGenerator()).printAvgAvgAbsCorr();

            for (int i = 0; i < paramsMeta.numPNegMethodsToUse; i++)
                for (int j = 0; j < paramsMeta.numClassifiers; j++)
                    toPrint += " " + trueRankingComparisons[i][j];
            toPrint += " " + lofRankingComparison;
            toPrint += " " + lofRankingComparison2;

            System.out.println(toPrint);
            writer.write(toPrint + "\n");
//            writer.write(((cade.dataGenerators.GaussianMixtureDataGenerator)params.getDataGenerator()).printGaussians());
        }
        writer.flush();
    }


}
