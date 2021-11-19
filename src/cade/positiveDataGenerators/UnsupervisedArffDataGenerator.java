package cade.positiveDataGenerators;

import java.io.IOException;
import java.util.Random;

import cade.Parameters;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Class for unsupervised framework, in which we're given an unlabeled data set and need to identify the outliers
 * within it. (Data file does have labels, but algorithms don't use them...
 * **Except possibly to control the amount of "contamination" in the training data.)
 *
 * test data: We'd like to always make predictions on every instance in the data set. However, if the "outlier" class
 *  is too common, we might downsample it here.
 * training data: a sample (of specified size) from the data file. (But actually, from the test data set, since this
 *  has adjusted the number of negatives.) A flag tells us whether to allow true negs in this set.
 *
 * [Was previously called ContaminatedArffDataGenerator. Under that setup, the amount of negatives in the test data
 * was always controlled with maxRatioNegativesToPositives, and whether these were also in the training data
 * was controlled with the flag allowNegsInTrainingData.]
 */
public class UnsupervisedArffDataGenerator extends LabeledArffDataGenerator {
    boolean limitPercentNegsInTestData = false;  // if false, use all the data's negatives in the test set
    double maxRatioNegativesToPositives = .02; // test data will include (this value * # of positives) negative instances
    boolean allowNegsInTrainingData = true;

    Instances positiveData, negativeData;   // corresponding to true labels
    Instances testingData;                  // the "whole data set". Consists of all positives and a percentage of negatives.
    Instances testNegatives, testPositives; // testNegatives U testPositives = testingData
    Instances trainingData;                 // What to give out as training data. A sample from either testingData, or if no contamination, only from the positives
    int numTrainingInstances;               // size of training set = min(maxNumInstances from constructor, testingData.numInstances())
                                            // Note: if this is set large (or to -1), training and testing sets will be identical.

    public UnsupervisedArffDataGenerator(Parameters params, String dataFile, int classAttr,
                                         String[] positiveClasses, String[] negativeClasses,
                                         int[] attributesToUse, int numTrainingInstances, Random rng) throws IOException {
        System.out.println("creatinng data generator");
	    this.dataFile = dataFile;
        this.classAttr = classAttr;
        this.positiveClasses = positiveClasses;
        this.negativeClasses = negativeClasses;
        this.attributesToUse = attributesToUse;
        this.numTrainingInstances = numTrainingInstances;
        this.deleteInstancesWithMissingValues = params.skipInstancesWithMissingVals;
        if (rng != null)
            randomness = rng;
        else randomness = new Random();
        initialize();
    }

    public void initialize() throws IOException {
        super.initialize();
        createPosNegSets();
        prepareForNextRun();
	    System.out.println("setting trainingSample");
        trainingSample = testingData; //this specifies which data to use when calculating data ranges
        calculatePropertiesFromData();
        numTrainingPositives = numTrainingInstances;    // restore numTrainPositives, because calculatePropertiesFromData() wrote to it
    }


    private void createPosNegSets() {
        //get the attributes from the data
        FastVector attributes = new FastVector(numAttributes + 1);
        for (int i = 0; i <= numAttributes; i++) {
            Attribute tempAttr = allData.attribute(i);
            attributes.addElement(tempAttr);
        }
        positiveData = new Instances("UCI data", attributes, 0);
        positiveData.setClassIndex(allData.classIndex());
        negativeData = new Instances("UCI data", attributes, 0);
        negativeData.setClassIndex(allData.classIndex());
        testingData = new Instances("UCI data", attributes, 0);
        testingData.setClassIndex(allData.classIndex());
        trainingData = new Instances("UCI data", attributes, 0);
        trainingData.setClassIndex(allData.classIndex());
        testNegatives = new Instances("UCI data", attributes, 0);
        testNegatives.setClassIndex(allData.classIndex());
        testPositives = new Instances("UCI data", attributes, 0);
        testPositives.setClassIndex(allData.classIndex());

        //separate the positive and negative data
        for (int i = 0; i < allData.numInstances(); i++) {
            Instance inst = allData.instance(i);
            if (allData.classAttribute().value((int) inst.classValue()).equals(Parameters.negativeClassLabel))
                negativeData.add(inst);
            else if (allData.classAttribute().value((int) inst.classValue()).equals(Parameters.positiveClassLabel)) {
                positiveData.add(inst);
//                testingData.add(inst);
            }
        }
    }

    public void prepareForNextRun() {
        positiveData.randomize(randomness);
        negativeData.randomize(randomness);

        // Construct test data: all positives, and some or all negatives
        testingData.delete();
        testNegatives.delete();
        testPositives.delete();
        for (int i = 0; i < positiveData.numInstances(); i++)
            testingData.add(positiveData.instance(i));

        int numNegsToAdd;
        if (limitPercentNegsInTestData) {
            numNegsToAdd = (int) (positiveData.numInstances() * maxRatioNegativesToPositives);
            if (numNegsToAdd < 5) {
                numNegsToAdd = 5;
            }
        } else {
            numNegsToAdd = negativeData.numInstances();
        }
        for (int i = 0; i < numNegsToAdd && i < negativeData.numInstances(); i++)
            testingData.add(negativeData.instance(i));

        //separate into positive and negative test data for when we retrieve it
        for (int i = 0; i < testingData.numInstances(); i++) {
            Instance inst = testingData.instance(i);
            if (testingData.classAttribute().value((int) inst.classValue()).equals(Parameters.negativeClassLabel))
                testNegatives.add(inst);
            else if (testingData.classAttribute().value((int) inst.classValue()).equals(Parameters.positiveClassLabel)) {
                testPositives.add(inst);
            }
        }

        if (numTrainingInstances <= 0 || numTrainingInstances > testingData.numInstances()) {
            numTrainingInstances = testingData.numInstances();
        }

        // Construct training data: some positives, and maybe some negatives.
        // Size limited by numTrainingInstances.
        testingData.randomize(randomness);

        trainingData.delete();
        //create training sample, labeling everything as positive
        int maxToGenerate = testingData.numInstances();
        if (!allowNegsInTrainingData)
            maxToGenerate = positiveData.numInstances();
        for (int i = 0; i < numTrainingInstances && i < maxToGenerate; i++) {
            if (allowNegsInTrainingData)
                trainingData.add(testingData.instance(i));
            else
                trainingData.add(positiveData.instance(i));
            //make sure it's positive
            trainingData.instance(trainingData.numInstances() - 1).setClassValue(Parameters.positiveClassLabel);
        }

    }


    //not actually guaranteed to be all true positives, but all are labeled positive
    public Instances generateTrainingPositives(int numInstances) {
        return trainingData;
    }

    public Instances generateTrainingNegatives(int numInstances) {
        return null;
    }

    //don't return testing positives and negatives separately
    public Instances generateTestingPositives(int numInstances) {
        return testPositives;
    }

    public Instances generateTestingNegatives(int numInstances) {
        return testNegatives;
    }


}
