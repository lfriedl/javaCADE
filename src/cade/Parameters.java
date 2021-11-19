package cade;

import cade.positiveDataGenerators.*;
import cade.generativeProbDensities.*;
import com.github.rcaller.rstuff.RCaller;
import com.github.rcaller.rstuff.RCode;

import java.io.IOException;
import java.util.Random;


/**
 * The Parameters class takes care of most configurable settings.
 * In particular:
 * -The settingNum for Params encodes the number of training examples, plus other things we rarely change.
 * -ParamsMultiRuns is where we store the lists of classifier & density estimate types we're going to
 * loop over (using a separate settingNum). Parameters stores a ParamsMultiRun object (asks for it in the constructor).
 * We use that object to find out which classifier(s), baseline classifier(s), and LOF method(s) to run.
 *
 * -Parameters holds a single dataGenerator object (created in the constructor).
 * To prepare it for the next fold (of cross-validation, etc.), call this.prepareDataGenForNextRun(). It sets up
 * the new set of training & test instances.
 * -Parameters sets up one pNeg method at a time. We do this with this.setPseudoNegTypeFromIndex() (which looks up the
 * index in paramsMeta arrays).
 * Then when it's time to generate pNegs, call this.getPseudoNegativeGenerator().
 * (This part is a bit complex, due to us allowing the "probDensity" not to match the "pNeg type".)
 * <p>
 * Our experiment code generally runs as follows:
 * -load dataset                                // single dataset passed in an arg to the driver;
 *                                              // it's sampled / subdivided into folds once and for all
 * for each pNeg method:                        // loop over pNeg methods & folds
 *      for each trial or fold:
 *          -generate positives
 *          -generate pNegs
 *
 *          for each classifier:                // loop over classifiers (using same training & test data)
 *              -train classifier
 *              -load test data and evaluate
 */
public class Parameters {

    // -- Constants defined here --
    public enum ClassifierType {RANDOMFOREST, KNN, LOGISTIC, TREE, NAIVEBAYES, NOCLASSIFIER;}
    public enum PseudoNegGenerationType {UNIFORM, MARGINAL_KDE, BAYESNET, GAUSSIAN, MIXTURE, ONE_EVERYWHERE}

    //POS_RANGE = range of true positives, FIVE_STDDEVS_FROM_MEAN = mean +/- 5*pos_stdDev, TWO_STDDEVS_FROM_MINMAX = min/max +/- 2*stdDev
    public enum UniformPNegRange {POS_RANGE, FIVE_STDDEVS_FROM_MEAN, TWO_STDDEVS_FROM_MINMAX}

    public enum TrueDataGenerationType {GAUSSIANMIXTURE, REALDATA, MULTIVARNORM}
    public static final String positiveClassLabel = "positive";
    public static final String negativeClassLabel = "negative";
    // -- End constants --


    // --- Fields describing what a run looks like. ---

    // todo: turn this to false when running it for real
    public static boolean runInRepeatableMode = true; // for debugging: use a fixed random seed
    public static int randomSeed = 1;

    // -- Fields that are commonly changed (via setup methods) --
    protected boolean runUnsupervisedMode = false;  // false -> our more-usual CrossVal mode. true -> use same data file, but treat it differently.

    // best performance when smoothToRemove1sAnd0s is true, but turn it off to countZeroes accurately (for diagnostics).
    public boolean smoothToRemove1sAnd0s;
    public boolean countZeroes;  // to print them out, driver must call paramsMeta.printSummaryOfPredictions(), then
                                 // paramsMeta.resetCountsOfZeroes() when finished with each pNeg method
    // -- End commonly changed fields --


    // -- Fields that are less commonly changed, with their defaults (ok to omit later, if unchanged) --
    protected TrueDataGenerationType trueDataType = TrueDataGenerationType.REALDATA;  // almost always
    public UniformPNegRange uniformPNegRange = UniformPNegRange.POS_RANGE; // note: with synthetic data, must use UniformPNegRange.FIVE_STDDEVS_FROM_MEAN

    public int numTrainPositives = -1;  // means "all of them" (subdivided for cross-validation); but sometimes overriden in the constructor
    public int numPseudoNegs = -1;      // i.e., "match the number of positives"
    public int numTestPositives = -1;   // "all the test data you have"
    public int numTestNegatives = -1;   // "all the test data you have"

    public boolean useAllNegatives;//if this is true, makes sure that we use all the negative instances and only limit the number of positives
    // (necessary for Vegas, where there are very few true negatives)

    public double mixtureFractionUniform = .01;  // applies (only) to mixture pNegs
    public boolean normalize = false;    //if set to true, all continuous features are normalized to be (val-mean)/stdev. (Only implemented in CrossValArffDataGenerator.)
    protected boolean replicateOldResults = false;  // if true, code is less efficient, but controlled random seed makes exact comparisons possible to runs from the paper.

    // Almost obsolete: these should generally be set to true
    public boolean useIntegerUniformPNegs = true;   // distinguish between integer & real-valued attrs, even though Weka doesn't
    public boolean skipInstancesWithMissingVals = true;  // i.e., delete those instances. when false, instances with missing values might silently cause AUC = .5.


    // Fields that are set and accessed internally
    public ParamsMultiRuns paramsMeta;

    // Note: a given instance of Parameters corresponds to just one data set, so
    // dataGenerator is created once and for all. In contrast, we vary the pseudoNegative types, making a new
    // ProbDensity whenever needed.
    protected DataGenerator dataGenerator; //reads data from file when created
    protected PseudoNegGenerationType pseudoNegType;
    protected PseudoNegGenerationType probDensityType;
    protected GenerativeProbDensity pNegGenerator = null;

    public static RCaller rCaller;
    public static RCode rCode;


    // Different constructors are used for different types of DataGenerators
    public Parameters() {   // stub allowing us to move obsolete stuff into child class
    }

    // These two are used in SyntheticDataExperiments
    public Parameters(int settingNumForRuns, ParamsMultiRuns paramsMeta, String covMethod) {
        this.paramsMeta = paramsMeta;
        trueDataType = TrueDataGenerationType.GAUSSIANMIXTURE;
        setUpVariablesDescribingOneRun(settingNumForRuns);
        dataGenerator = new GaussianMixtureDataGenerator(5, numTrainPositives, covMethod);
    }

    public Parameters(int settingNumForRuns, ParamsMultiRuns paramsMeta, int numDimensions, int numNonsenseDims) {
        this.paramsMeta = paramsMeta;
        trueDataType = TrueDataGenerationType.GAUSSIANMIXTURE;
        setUpVariablesDescribingOneRun(settingNumForRuns);
        dataGenerator = new GaussianMixtureDataGenerator(numDimensions, numNonsenseDims, numTrainPositives);
    }

    // For labeled data files (e.g., UCI data). Won't work with synthetic data.
    // numInstancesInCrossValDataset = -1 means use all the instances available
    public Parameters(String dataFile, int classAttr, String[] positiveClasses, String[] negativeClasses,
                      int[] attributesToUse, int numInstancesInCrossValDataset, int settingNumForRuns,
                      ParamsMultiRuns paramsMeta) throws IOException {
        this.paramsMeta = paramsMeta;
        setUpVariablesDescribingOneRun(settingNumForRuns);
        if (runUnsupervisedMode)
            dataGenerator = new UnsupervisedArffDataGenerator(this, dataFile, classAttr, positiveClasses,
                    negativeClasses, attributesToUse, numTrainPositives, runInRepeatableMode ? new Random(randomSeed) : null);
        else
            dataGenerator = new CrossValArffDataGenerator(this, dataFile, classAttr, positiveClasses,
                    negativeClasses, attributesToUse, numInstancesInCrossValDataset, runInRepeatableMode ? new Random(randomSeed) : null);
    }

    // For unlabeled data files
    public Parameters(String dataFile, int[] attributesToUse, int[] idAttributes, int numTrainingInstances, int settingNumForRuns, ParamsMultiRuns paramsMeta) throws IOException {
        this.paramsMeta = paramsMeta;
        setUpVariablesDescribingOneRun(settingNumForRuns);
        dataGenerator = new NoLabelsArffDataGenerator(dataFile, attributesToUse, idAttributes, numTrainingInstances,
                runInRepeatableMode ? new Random(randomSeed) : null, skipInstancesWithMissingVals);
    }


    protected void setUpVariablesDescribingOneRun(int settingNum) {
        // Note: for synthetic data, UniformPNegRange can't be POS_RANGE or TWO_STDDEVS_FROM_MINMAX  (doesn't know min/max)

        // space up here: play with vars as desired, using settingNums we haven't codified yet

        if (settingNum == 21) { // KEEP. Used in runCrossValWithLabeledData(), sometimes, I think.
            // unsupervised mode
            runUnsupervisedMode = true;
            numTrainPositives = 10000;  // this is the max we'll use. If data file is smaller, we just train on all of it.

            smoothToRemove1sAnd0s = false;
            countZeroes = true;     // only affects what's printed out
        }
        if (settingNum == 20) {  // KEEP. Used in runCrossValWithLabeledData()
            // post-refactoring: what is still needed?
            numTrainPositives = 10000; // i.e., all of them (will be subdivided during cross-validation)

            smoothToRemove1sAnd0s = true;
            countZeroes = false;     // only affects what's printed out

        }
        if (settingNum == 19) {  // KEEP. Used in runUnlabeledWithCommandLineArgs()
            // unlabeled data -- updated to use smoothing and print out zeroes info

            // Data generator sets the equivalent of numTrainPositives in its constructor, and
            // we have hard-coded elsewhere to grab "an equal number of pseudoNegs" and then "all the test data you have".
            numTrainPositives = -1;   // I.e., use what we specify when we set up the data generator
            numPseudoNegs = -1;  // Code for "whatever the positives were"

            smoothToRemove1sAnd0s = true;
//            skipInstancesWithMissingVals = true;
            countZeroes = true;
        }
        if (settingNum == 18) {  // KEEP. Used in BayesNetTest.
            //Most current HempstalkCorrection. New thing: remove instances with missing values
            //can't change - used in a test!!
            smoothToRemove1sAnd0s = true;
            skipInstancesWithMissingVals = true;

            countZeroes = false;     // only affects what's printed out
        }
        //Used with Gaussian mixture model, in set-up to randomly generate positives, train against uniform pNegs, and save ranking with PAD/naive density estimate
        if (settingNum == 17) {  // KEEP. Used in SyntheticDataExperiments.
            numTrainPositives = 1000;
            numPseudoNegs = 1000;
            numTestPositives = 1000;
            numTestNegatives = 1000;
            uniformPNegRange = UniformPNegRange.POS_RANGE;
            trueDataType = TrueDataGenerationType.GAUSSIANMIXTURE;
//            useIntegerUniformPNegs = true;//shouldn't actually apply - they're not integers
            smoothToRemove1sAnd0s = true;
        }
        if (settingNum == 13) {  // KEEP. Used in NoClassifierTest.
            // most current Hempstalk setting.
            smoothToRemove1sAnd0s = false;
        } else if (settingNum == 9) {  // KEEP. Used throughout HempstalkTest.java,
            // also UniformPseudoNegativeGeneratorTest and ArffDataPNegGeneratorTest.
            // What we use for UCI data runs with no box. Behavior not allowed to change, since it's used in
            // test cases.
            useIntegerUniformPNegs = false;
            skipInstancesWithMissingVals = false;    // the only time it ever makes sense to do this: to preserve behavior in tests

        } else if (settingNum == 8) {
            // What we use for UCI data runs with no box. Behavior not allowed to change, since it's used in
            // test cases.
            useIntegerUniformPNegs = false;
            skipInstancesWithMissingVals = true;
        }


        System.out.println("Smoothing is " + ((smoothToRemove1sAnd0s) ? "on" : "off"));
        System.out.println("Missing value handling is " + ((skipInstancesWithMissingVals) ? "on" : "off"));
        System.out.println("Running in " + ((runUnsupervisedMode) ? "unsupervised" : "cross-validated") + " mode");
        if (runInRepeatableMode) {
            System.out.println("Running in repeatable mode (with a controlled random seed)");
        }


        // Adding these to Parameters so that we only need run 1 R process (much faster).
        rCaller = RCaller.create();
        rCode = RCode.create();
        rCaller.setRCode(rCode);

    }

    // data generator methods

    public void prepareDataGenForNextRun() {
        dataGenerator.prepareForNextRun();
    }

    public DataGenerator getDataGenerator() {
        return dataGenerator;
    }

    // pseudo negative generator methods

    public void setPseudoNegTypeFromIndex(int pNegIndex) {
        pNegGenerator = null;   // since this is the call to reset them
        this.pseudoNegType = paramsMeta.pseudoNegGenerationTypesThisRun[pNegIndex];
        if (paramsMeta.useSameProbDensityMethods)
            this.probDensityType = this.pseudoNegType;
        else
            this.probDensityType = paramsMeta.probDensityTypesThisRun[pNegIndex];

    }

    // pNeg density estimator. In initial implementation, all three methods below were identical,
    // and they returned a brand new, initialized object.
    // However, calling any of these may be expensive; depending on the class, it may *learn* an estimator (fit it to
    // the training data). So, once you've created it, store it for later use.
    // Creates and initializes a PseudoNegativeGenerator from the current training data
    public GenerativeProbDensity getPseudoNegativeGenerator() {
        this.pNegGenerator = getNewProbDensity(pseudoNegType);
        if (runInRepeatableMode) {
            pNegGenerator.setRandom(new Random(randomSeed));
//            System.out.println("Running in repeatable mode (with a controlled random seed)");
        }
        return pNegGenerator;
    }

    // Almost always, this just means the (trained) pNegGenerator again. Except for a special case...
    public GenerativeProbDensity getProbDensity() {
        if (paramsMeta.useSameProbDensityMethods && !(replicateOldResults && runInRepeatableMode))
            return pNegGenerator;
        else
            return getNewProbDensity(probDensityType);
    }

    // The class to create depends on the TrueDataGenerationType and the PseudoNegGeneration/ProbDensity Type
    private GenerativeProbDensity getNewProbDensity(PseudoNegGenerationType pDensType) {
        if (pDensType == PseudoNegGenerationType.MIXTURE) {
            return new MixurePNegGenerator(this, mixtureFractionUniform);
        } else if (pDensType == PseudoNegGenerationType.UNIFORM) {
            return new UniformProbDensity(this);
        } else if (pDensType == PseudoNegGenerationType.ONE_EVERYWHERE) {
            return new OneEverywhereProbDensity(this);
        } else if (pDensType == PseudoNegGenerationType.BAYESNET) {
            return new BayesNetProbDensity(this);
        } else if (pDensType == PseudoNegGenerationType.GAUSSIAN) {
            return new GaussianProbDensity(this);
        } else {
            return new IndepMarginalsProbDensity(this);
        }
    }

}
