package cade.positiveDataGenerators;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashSet;

import weka.core.Instance;
import weka.core.Instances;

public abstract class DatafileDataGenerator extends DataGenerator {
    protected String[] positiveClasses, negativeClasses;
    protected int[] attributesToUse;
    protected String dataFile;

//    public Instances getAllDataWithLabels() {
//        return allData;
//    }

    protected Instances allData;  // includes class label
    protected boolean deleteInstancesWithMissingValues;

    public void loadData() throws IOException {
        BufferedReader reader;

        reader = new BufferedReader(new FileReader(dataFile));
        allData = new Instances(reader);
        reader.close();
        System.out.println("Read data from " + dataFile);

        if (deleteInstancesWithMissingValues) {
            System.err.println("Removing any instances with missing values from the data set");
            Instances noMissing = new Instances(allData, allData.numInstances()); // create empty copy
            for (int i = 0; i < allData.numInstances(); i++) {
                if (! allData.instance(i).hasMissingValue()) {
                    noMissing.add(allData.instance(i));
                }
            }
            allData = noMissing;
        }

    }

    public Instance getOneExampleInstance() {
        return allData.firstInstance();
    }

    // Calculates properties that our Pseudo-negative generators will need, using current
    // value of trainingSample.
    public void calculatePropertiesFromData() {

        min = new double[numAttributes];
        max = new double[numAttributes];
        valuesSeen = new HashSet[numAttributes];

        double[] sums = new double[numAttributes];
        double[] sumOfSquares = new double[numAttributes];

        posMeans = new double[numAttributes];
        posStdDevs = new double[numAttributes];

        for (int j = 0; j < numAttributes; j++) {
            if (trainingSample.attribute(j).isNumeric()) {//only applies for numeric data
                // initialize min & max
                min[j] = trainingSample.instance(0).value(j);
                max[j] = trainingSample.instance(0).value(j);

                // Loop over instances
                for (int i = 0; i < trainingSample.numInstances(); i++) {
                    Instance inst = trainingSample.instance(i);

                    //calculate sum of squares
                    sums[j] += inst.value(j);
                    sumOfSquares[j] += Math.pow(inst.value(j), 2);

                    //calculate min and max
                    if (inst.value(j) < min[j])
                        min[j] = inst.value(j);
                    else if (inst.value(j) > max[j])
                        max[j] = inst.value(j);
                }

                // Calculate summaries per attribute
                posMeans[j] = sums[j] / trainingSample.numInstances();
                double variance = (sumOfSquares[j] / trainingSample.numInstances()) - Math.pow(posMeans[j], 2);
                posStdDevs[j] = Math.sqrt(variance);
            } else { //nominal, so get a list of what values are there
            	valuesSeen[j] = new HashSet<Integer>();
            	
            	//Loop over instances
            	for (int i = 0; i < trainingSample.numInstances(); i++){
            		Instance inst = trainingSample.instance(i);
            		
            		valuesSeen[j].add((int)inst.value(j));
            	}
            }
        }

        numTrainingPositives = trainingSample.numInstances();
    }

    //given the index of a class value, checks if its corresponding string value is in classes
    // note: can't use getOneExampleInstance() here because we're still loading data
    protected boolean valueInClass(double classValue, String[] classes, Instance exampleInstance){
    	String value = exampleInstance.classAttribute().value((int)classValue);
	   	for (int i=0; i<classes.length; i++)
    		if (value.equals(classes[i]))
    			return true;
    	return false;
    }

    protected void removeUnwantedAttributes(Instances someData, int[] attributesToUse) {

        if (attributesToUse.length != 0) {
            //now remove unused attributes
            for (int i = someData.numAttributes() - 1; i >= 0; i--) {
                //check if i's an attribute we're using
                boolean contains = false;
                for (int val : attributesToUse)
                    if (val - 1 == i || someData.classIndex() == i)
                        contains = true;
                if (!contains) {
                    someData.deleteAttributeAt(i);
                }
            }
        }
    }

}
