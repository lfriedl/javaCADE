package cade.positiveDataGenerators;

import java.io.IOException;
import java.util.Random;

import cade.Parameters;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class CrossValArffDataGenerator extends LabeledArffDataGenerator {
    Instances[] positiveData;
    Instances[] negativeData;
    Instances trainingSampleNegatives;      // complement to trainingSample field inherited from DatafileDataGenerator,
                                            // only ever used to run the classifier baseline

    int numInstancesUsedInDataSet; //number of data instances used from file = min(allData.numInstances, maxNumInstances wanted)
    boolean useAllNegatives;//if true, numInstancesUsedInDataSet limits only the number of positives. use all negatives (for testing)

    boolean useAllTestData; // train as usual, but make predictions on full data file [never used]
    Instances allPositiveData, allNegativeData;

    int numFolds;
    protected int currFold = -1;

    public CrossValArffDataGenerator(Parameters params, String dataFile, int classAttr,
                                     String[] positiveClasses, String[] negativeClasses,
                                     int[] attributesToUse, int maxNumInstances, Random rng) throws IOException {
        numFolds = params.paramsMeta.numRunsPerSetting;
        this.dataFile = dataFile;
        this.classAttr = classAttr;
        this.positiveClasses = positiveClasses;
        this.negativeClasses = negativeClasses;
        this.attributesToUse = attributesToUse;
        this.numInstancesUsedInDataSet = maxNumInstances;
        this.useAllNegatives = params.useAllNegatives;
        this.deleteInstancesWithMissingValues = params.skipInstancesWithMissingVals;
        this.normalize = params.normalize;
        if (rng != null)
            randomness = rng;
        else randomness = new Random();

        initialize();
    }

    public void initialize() throws IOException {

        super.initialize();
        divideIntoFolds();
        calculateCorrelation();
        prepareForNextRun();
    }


    private void divideIntoFolds() {

    	boolean numInstsSpecified = true;
        if (numInstancesUsedInDataSet == -1){
            numInstancesUsedInDataSet = allData.numInstances();
            numInstsSpecified = false;
        } else if (numInstancesUsedInDataSet > allData.numInstances())
        	numInstancesUsedInDataSet = allData.numInstances();
        
        int numPerFold = numInstancesUsedInDataSet / numFolds;

        //get the attributes from the data
        FastVector attributes = new FastVector(numAttributes + 1);
        for (int i = 0; i <= numAttributes; i++) {
            Attribute tempAttr = allData.attribute(i); // I assume we'll never need the same later
            attributes.addElement(tempAttr);
        }

        positiveData = new Instances[numFolds];
        negativeData = new Instances[numFolds];
       
        
        int numPositive = 0;
        int numNegative = 0;
        //only used if useAllNegatives is true (both used if useAllTestData is true)
        allNegativeData = new Instances("UCI data", attributes, 0);
        allPositiveData = new Instances("Positive data", attributes, 0);
        allNegativeData.setClassIndex(allData.classIndex());

        int index = 0; //the index of the instance be taken from data
        for (int i = 0; i < numFolds; i++) {
            positiveData[i] = new Instances("UCI data", attributes, 0);
            positiveData[i].setClassIndex(allData.classIndex());
            negativeData[i] = new Instances("UCI data", attributes, 0);
            negativeData[i].setClassIndex(allData.classIndex());
            
            for (int j = i * numPerFold; j < (i + 1) * numPerFold; j++) {
                Instance inst = allData.instance(index++);
                if (allData.classAttribute().value((int) inst.classValue()).equals(Parameters.negativeClassLabel)) {
                	if (useAllNegatives || useAllTestData){//divide it into folds later - just store/count it now
                		allNegativeData.add(inst);
                		//if we want to use exactly numInstances positives, we can't count this, so we must decrement j
                		if (numInstsSpecified)
                			j--;
                	} else
                		negativeData[i].add(inst);
                    numNegative++;
                } else {
                	if (useAllTestData)
                		allPositiveData.add(inst);
                    positiveData[i].add(inst);
                    numPositive++;
                }
            }
        }
        
        //now, if we're using all the negatives, go through and divide them into folds
        if (useAllNegatives){
        	//first, look at all the instances we didn't get to when dividing positives into folds
        	for (int i = numFolds*numPerFold; i < allData.numInstances(); i++){
        		Instance inst = allData.instance(i);
        		if (allData.classAttribute().value((int) inst.classValue()).equals(Parameters.negativeClassLabel))
        			 allNegativeData.add(inst);
        	}
        	
        	//now that we have all the negatives in allNegativeData, divide it into numFolds folds
        	int numNegsPerFold = allNegativeData.numInstances()/numFolds;
        	for (int i = 0; i < numFolds; i ++){
        		for (int j = i * numNegsPerFold; j < (i+1) * numNegsPerFold; j++){
        			Instance inst = allNegativeData.instance(j);
        			negativeData[i].add(inst);
        		}
        	}
        }
        System.out.println("num positives: " + numPositive);
        percentPositive = ((double) numPositive) / (numPositive + numNegative);

    }

    //formerly, this was setFold, which took a fold number and set stuff up
    public void prepareForNextRun() {
        currFold++;
        if (currFold >= numFolds)
        	currFold = 0;

        trainingSample = getTrainingPositivesForCurrFold();
        calculatePropertiesFromData();  // computes statistics of this.trainingSample

        trainingSampleNegatives = getTrainingNegativesForCurrFold();
    }

    public Instances getTrainingPositivesForCurrFold() {
        Instances newInstances = new Instances(positiveData[0], 0);

        for (int f = 0; f < numFolds; f++) {
            if (f == currFold)
                continue;
            Instances trainingData = positiveData[f];//current data being looked at
            for (int i = 0; i < trainingData.numInstances(); i++) {
                newInstances.add(trainingData.instance(i));
            }
        }
        return newInstances;
    }

    public Instances getTrainingNegativesForCurrFold() {

        Instances newInstances = new Instances(positiveData[0], 0);

        for (int f = 0; f < numFolds; f++) {
            if (f == currFold)
                continue;
            Instances trainingData = negativeData[f];
            for (int i = 0; i < trainingData.numInstances(); i++)
                newInstances.add(trainingData.instance(i));
        }

        return newInstances;
    }

    public Instances generateTrainingPositives(int numInstances) {
        return getEnoughInstances(trainingSample, numInstances);
    }

    public Instances generateTrainingNegatives(int numInstances) {
        return getEnoughInstances(trainingSampleNegatives, numInstances);
    }

    public Instances generateTestingPositives(int numInstances) {
    	if (useAllTestData)
    		return allPositiveData;
        return getEnoughInstances(positiveData[currFold], numInstances);
    }

    public Instances generateTestingNegatives(int numInstances) {
    	if (useAllTestData)
    		return allNegativeData;
        return getEnoughInstances(negativeData[currFold], numInstances);
    }

    public int numTrainingPositives() {
        return trainingSample.numInstances();
    }
}
