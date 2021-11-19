package cade;

import cade.classifiers.MultipleClassifierFDC;
import cade.lof.LOFWrapper;

import java.util.Random;

/**
 * Parameters needs one of these.
 */

public class ParamsMultiRuns {// Fields describing things across multiple runs
    // -- Fields that need to be specified --
    public int numRunsPerSetting;      // i.e., numFolds. Usually 10 for expts, 1 for an unlabeled data file.
    public boolean runLOFs = false;    // note: only runs if some variation of CADE does too

    public Parameters.PseudoNegGenerationType[] pseudoNegGenerationTypesThisRun = new Parameters.PseudoNegGenerationType[]{};
    public Parameters.ClassifierType[] classifierTypesThisRun = new Parameters.ClassifierType[]{};
    // (This class's 2 fields for baselines are a little messy.)
    // baselineClassifierTypesThisRun refers to maybe not the baseline you were thinking of: it trains & tests on pos vs. pNegs.
    // Useful for diagnostics, but *not* a ceiling.
    public Parameters.ClassifierType[] baselineClassifierTypesThisRun = new Parameters.ClassifierType[]{};
    // -- End of fields to specify --

    // Uncommon. Only change if you want a mismatched pNeg generation method and density estimation method.
    // This can be used for certain baselines, and for experiments.
    public boolean useSameProbDensityMethods = true;
    public Parameters.PseudoNegGenerationType[] probDensityTypesThisRun = new Parameters.PseudoNegGenerationType[]{};

    public boolean runTrueNegBaselineInstead = false;   // flag for if we're running the "normal" baseline instead; probably via BaselineExptMethods.classifyPosFromTrueNeg()?
                                                        // when true, makes sure to loop over only the classifiers, nothing else.

    // -- These are set upon initialization --
    public int numClassifiers; // total of regular and baselines, so we know how much array space to allocate
    public int numPNegMethodsToUse;
    protected StatsZeroCounter[] statsZeroCounters;


    public ParamsMultiRuns(int settingNum) {
        setupVariablesDescribingSetOfRuns(settingNum);
        initializeOtherFields();
    }

    // selects the classifier and pNeg method by their index in the enumerated list
    public ParamsMultiRuns(int classiferNumber, int pNegNumber) {
        classifierTypesThisRun = new Parameters.ClassifierType[]{
                Parameters.ClassifierType.values()[classiferNumber]};
        pseudoNegGenerationTypesThisRun = new Parameters.PseudoNegGenerationType[]{
                Parameters.PseudoNegGenerationType.values()[pNegNumber]};
        baselineClassifierTypesThisRun = new Parameters.ClassifierType[]{};

        initializeOtherFields();
    }

    protected void initializeOtherFields() {
        numClassifiers = classifierTypesThisRun.length + baselineClassifierTypesThisRun.length;
        numPNegMethodsToUse = pseudoNegGenerationTypesThisRun.length;
        statsZeroCounters = new StatsZeroCounter[numClassifiers];
        for (int i = 0; i < numClassifiers; i++) {
            statsZeroCounters[i] = new StatsZeroCounter();
        }

        if (runTrueNegBaselineInstead) {
            numClassifiers = classifierTypesThisRun.length; // no additional baselines
            pseudoNegGenerationTypesThisRun = new Parameters.PseudoNegGenerationType[]{Parameters.PseudoNegGenerationType.UNIFORM};
            // really, there are 0 pseudoNegGenerationTypesThisRun. But this line means: "store one result."
        }
    }

    protected void setupVariablesDescribingSetOfRuns(int settingNum) {

        if (settingNum == 17) { // New setting in 2021. Some demos.
//            pseudoNegGenerationTypesThisRun = new Parameters.PseudoNegGenerationType[]{Parameters.PseudoNegGenerationType.UNIFORM, Parameters.PseudoNegGenerationType.GAUSSIAN, Parameters.PseudoNegGenerationType.MARGINAL_KDE, Parameters.PseudoNegGenerationType.BAYESNET};
            pseudoNegGenerationTypesThisRun = new Parameters.PseudoNegGenerationType[]{Parameters.PseudoNegGenerationType.MARGINAL_KDE, Parameters.PseudoNegGenerationType.BAYESNET};
//            classifierTypesThisRun = new Parameters.ClassifierType[]{Parameters.ClassifierType.NOCLASSIFIER};
            classifierTypesThisRun = new Parameters.ClassifierType[]{Parameters.ClassifierType.NOCLASSIFIER, Parameters.ClassifierType.KNN, Parameters.ClassifierType.RANDOMFOREST};
            numRunsPerSetting = 10;
            runLOFs = true;
        }


        if (settingNum == 15) {
            // Reserving this setting for (various) LOF runs
            numRunsPerSetting = 10;
            runLOFs = true;

            pseudoNegGenerationTypesThisRun = new Parameters.PseudoNegGenerationType[]{Parameters.PseudoNegGenerationType.UNIFORM};
            classifierTypesThisRun = new Parameters.ClassifierType[]{Parameters.ClassifierType.NOCLASSIFIER};
        }

        if (settingNum == 14) {  // KEEP. Used in runGaussianMixtureExperiments()
            numRunsPerSetting = 10;
            runLOFs = true;
            pseudoNegGenerationTypesThisRun = new Parameters.PseudoNegGenerationType[]{Parameters.PseudoNegGenerationType.UNIFORM, Parameters.PseudoNegGenerationType.GAUSSIAN, Parameters.PseudoNegGenerationType.MARGINAL_KDE};
            classifierTypesThisRun = new Parameters.ClassifierType[]{Parameters.ClassifierType.NAIVEBAYES, Parameters.ClassifierType.KNN, Parameters.ClassifierType.LOGISTIC, Parameters.ClassifierType.TREE, Parameters.ClassifierType.RANDOMFOREST, Parameters.ClassifierType.NOCLASSIFIER};
        }

        // "Box" only (uniform kind); no classifier
        if (settingNum == 13) {
            numRunsPerSetting = 10;
            pseudoNegGenerationTypesThisRun = new Parameters.PseudoNegGenerationType[]{Parameters.PseudoNegGenerationType.UNIFORM};
            classifierTypesThisRun = new Parameters.ClassifierType[]{Parameters.ClassifierType.NOCLASSIFIER};
            baselineClassifierTypesThisRun = new Parameters.ClassifierType[]{};
        }

        // mixture pNeg distribution
        if (settingNum == 12) {
            numRunsPerSetting = 10;
            pseudoNegGenerationTypesThisRun = new Parameters.PseudoNegGenerationType[]{Parameters.PseudoNegGenerationType.MIXTURE};
            classifierTypesThisRun = new Parameters.ClassifierType[]{Parameters.ClassifierType.NAIVEBAYES, Parameters.ClassifierType.KNN, Parameters.ClassifierType.LOGISTIC, Parameters.ClassifierType.TREE, Parameters.ClassifierType.RANDOMFOREST};
            baselineClassifierTypesThisRun = new Parameters.ClassifierType[]{};
        }


        // UCI data, but only run marginals. Or only run uniforms. Or mess with what you want. (Experimental)
        if (settingNum == 10) {  // KEEP. Used everywhere! In runCrossValWithLabeledData().
            numRunsPerSetting = 10;
            runLOFs = false;

            // These lines demonstrate how to get the classifier output alone, after training with arbitrary pNegs.
            // It gives 4 combos: no classifier + Gaussian pNegs, RF + Gaussian, classifier output only of noclassifier
            // + gaussian pNegs (always .5 auc), classifier output only of RF + gaussian pNegs.
//            pseudoNegGenerationTypesThisRun = new Parameters.PseudoNegGenerationType[]{Parameters.PseudoNegGenerationType.GAUSSIAN, Parameters.PseudoNegGenerationType.GAUSSIAN};
//            useSameProbDensityMethods = false;
//            probDensityTypesThisRun = new Parameters.PseudoNegGenerationType[]{Parameters.PseudoNegGenerationType.GAUSSIAN, Parameters.PseudoNegGenerationType.ONE_EVERYWHERE};
//            classifierTypesThisRun = new Parameters.ClassifierType[]{Parameters.ClassifierType.NOCLASSIFIER, Parameters.ClassifierType.RANDOMFOREST};
//            baselineClassifierTypesThisRun = new Parameters.ClassifierType[]{};

//            baselineClassifierTypesThisRun = new Parameters.ClassifierType[]{Parameters.ClassifierType.NOCLASSIFIER, Parameters.ClassifierType.RANDOMFOREST};
//            useSameProbDensityMethods = false;
//            probDensityTypesThisRun = new Parameters.PseudoNegGenerationType[]{Parameters.PseudoNegGenerationType.ONE_EVERYWHERE, Parameters.PseudoNegGenerationType.ONE_EVERYWHERE};
//            pseudoNegGenerationTypesThisRun = new Parameters.PseudoNegGenerationType[]{Parameters.PseudoNegGenerationType.MARGINAL};

            // This setting is the whole slew we used in the paper
            useSameProbDensityMethods = true;
            pseudoNegGenerationTypesThisRun = new Parameters.PseudoNegGenerationType[]{Parameters.PseudoNegGenerationType.UNIFORM, Parameters.PseudoNegGenerationType.GAUSSIAN, Parameters.PseudoNegGenerationType.MARGINAL_KDE, Parameters.PseudoNegGenerationType.BAYESNET};
            classifierTypesThisRun = new Parameters.ClassifierType[]{Parameters.ClassifierType.NAIVEBAYES, Parameters.ClassifierType.KNN, Parameters.ClassifierType.LOGISTIC, Parameters.ClassifierType.TREE, Parameters.ClassifierType.RANDOMFOREST, Parameters.ClassifierType.NOCLASSIFIER};
        }


        //Used with UCI data
        else if (settingNum == 8) {  // KEEP. Used throughout HempstalkTest and ArffDataPNegGeneratorTest
            // and UniformPseudoNegativeGeneratorTest and SpecificValueTest.
            numRunsPerSetting = 10;
            pseudoNegGenerationTypesThisRun = new Parameters.PseudoNegGenerationType[]{Parameters.PseudoNegGenerationType.MARGINAL_KDE};
            classifierTypesThisRun = new Parameters.ClassifierType[]{Parameters.ClassifierType.NOCLASSIFIER, Parameters.ClassifierType.RANDOMFOREST};//Parameters.getAllClassifierTypes();//new Parameters.ClassifierType[]{Parameters.ClassifierType.NAIVEBAYES};//
            baselineClassifierTypesThisRun = new Parameters.ClassifierType[]{};//

        }

    }

    // Gets the MultipleClassifierFDC (previously, LocalClassifier) for this run's classifier types.
    public MultipleClassifierFDC getClassifier() {
        return new MultipleClassifierFDC(classifierTypesThisRun, statsZeroCounters);
    }

    public MultipleClassifierFDC getBaselineClassifier() {
        return new MultipleClassifierFDC(baselineClassifierTypesThisRun, statsZeroCounters);
    }

    // LOF doesn't depend on pNegs, so only ever run it while doing the first pNeg method of a set
    public LOFWrapper getLOFWrapper(int pNegIndex, Random randomGenerator) {
        if (pNegIndex == 0 && runLOFs)
            return new LOFWrapper(randomGenerator);
        else return null;
    }


    public Parameters.ClassifierType[] getAllClassifierTypesThisRun() {
        Parameters.ClassifierType[] allClassifiersRun = new Parameters.ClassifierType[numClassifiers];
        for (int i = 0; i < classifierTypesThisRun.length; i++) {
            allClassifiersRun[i] = classifierTypesThisRun[i];
        }
        for (int i = 0; i < baselineClassifierTypesThisRun.length; i++) {
            allClassifiersRun[i + classifierTypesThisRun.length] = baselineClassifierTypesThisRun[i];
        }
        return allClassifiersRun;
    }

    public void printSummaryOfPredictions() {
        if (statsZeroCounters[0] != null)
            statsZeroCounters[0].printSummaryOfPredictions();
        for (int i = 0; i < statsZeroCounters.length; i++) {
            statsZeroCounters[i].printOneClassifierSummary(i);
        }
    }

    public void resetCountsOfZeroes() {
        statsZeroCounters = new StatsZeroCounter[numClassifiers];
        for (int i = 0; i < numClassifiers; i++) {
            statsZeroCounters[i] = new StatsZeroCounter();
        }
    }

}
