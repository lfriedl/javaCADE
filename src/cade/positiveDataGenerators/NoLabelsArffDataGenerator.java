package cade.positiveDataGenerators;

import java.io.IOException;
import java.util.Random;

import cade.Parameters;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 * For situation where data isn't labeled, but instead identified with IDs.
 * We'll sample a training set, and "test" (i.e., apply the method) to all data
 */
public class NoLabelsArffDataGenerator extends DatafileDataGenerator {

    public int[] idAttributes;//indexed from 1
    public String[][] idAttrValues;
    protected Attribute[] savedIDAttrs;


    public NoLabelsArffDataGenerator(String dataFile, int[] attributesToUse, int[] idAttributes,
                                     int numTrainPositives, Random rng, boolean ignoreInstancesWithMissingValues) throws IOException {
        this.dataFile = dataFile;
        this.attributesToUse = attributesToUse;
        this.idAttributes = idAttributes;
        this.numTrainingPositives = numTrainPositives;

        if (rng != null)
            randomness = rng;
        else randomness = new Random();

        this.deleteInstancesWithMissingValues = ignoreInstancesWithMissingValues;

        initialize();
    }


    private void initialize() throws IOException {
        loadData();
        if (numTrainingPositives == -1)
        	numTrainingPositives = allData.numInstances();

        if (idAttributes[0] != -1)
        	storeIDsForLater();

        // Make the attributesToUse array be explicit, because even if it was passed in empty, we want the id attributes to go away
        if (idAttributes[0] != -1)
        	buildAttrsToUse();
        removeUnwantedAttributes(allData, attributesToUse);

        // Do this after removing ID attrs, so that we can ignore unwanted class attrs by treating them as IDs
        setUpClassLabel(allData);
        percentPositive = 100;

        //numAttributes doesn't include class label
        numAttributes = allData.numAttributes() - 1;

        System.out.println("Preparing training sample");
	
	// old: calculates properties only from training sample	
        //trainingSample = getEnoughInstances(allData, numTrainingPositives);
        //calculatePropertiesFromData(); // computes statistics of this.trainingSample
// todo: double check whether/why this is really ok
	// new: calculate properties (data ranges) from entire data set, even if we use only a subset for training
        trainingSample = allData;
        int storeNumTrainingPositives = numTrainingPositives; // save it, because calculatePropertiesFromData() writes to this.numTrainingPositives
        calculatePropertiesFromData(); // computes statistics of this.trainingSample
        // restore variables
        numTrainingPositives = storeNumTrainingPositives;
        trainingSample = getEnoughInstances(allData, numTrainingPositives);

        System.out.println("Training sample contains " + trainingSample.numInstances() + " instances");


    }

    // Makes attributesToUse explicit
    private void buildAttrsToUse() {

        // Note: When it's passed explicitly, it won't contain ID attributes, so it's fine as is.
        if (attributesToUse.length > 0)
            return;

        // attributesToUse always remains indexed by 1.
        int indexInNewAttrsToUse = 0;
        attributesToUse = new int[allData.numAttributes() - idAttributes.length];
        for (int attrNum1Based = 1; attrNum1Based <= allData.numAttributes(); attrNum1Based++) {
            boolean isClassAttr = false;
            for (int idAttrNum : idAttributes) {
                if (idAttrNum == attrNum1Based)
                    isClassAttr = true;
            }
            if (!isClassAttr) {
                attributesToUse[indexInNewAttrsToUse] = attrNum1Based;
                indexInNewAttrsToUse++;
            }
        }
    }

    private void storeIDsForLater() {
        idAttrValues = new String[idAttributes.length][allData.numInstances()];

        int idIndex = 0;  // in idAttrValues

        for (int attr1Based: idAttributes) {  // attr is still 1-based
            int attr = attr1Based - 1;
            if (!allData.attribute(attr).isString() && !allData.attribute(attr).isNominal()) {
                for (int inst = 0; inst < allData.numInstances(); inst++)
                    idAttrValues[idIndex][inst] = "" + allData.instance(inst).value(attr);
            }
            else {
                for (int inst = 0; inst < allData.numInstances(); inst++)
                    idAttrValues[idIndex][inst] = allData.attribute(attr).value((int) allData.instance(inst).value(attr));
            }
            idIndex++;
        }

        // possibly a better way to save these for later:
        savedIDAttrs = new Attribute[idAttributes.length];
        for (int i = 0; i < idAttributes.length; i++) {
            savedIDAttrs[i] = (Attribute) allData.attribute(idAttributes[i]-1).copy();   // idAttributes is 1-based
        }
        // (then we can just paste them back in, later)

    }

    //data has no class label, so add a new column with the positive class label
    protected void setUpClassLabel(Instances data) {
        FastVector classVals = new FastVector(2);
        classVals.addElement(Parameters.positiveClassLabel);
        classVals.addElement(Parameters.negativeClassLabel);
        Attribute classAttr = new Attribute("classAttr", classVals);
        data.insertAttributeAt(classAttr, data.numAttributes());

        for (int i = 0; i < data.numInstances(); i++) {
            Instance inst = data.instance(i);
            inst.setValue(data.numAttributes() - 1, Parameters.positiveClassLabel);
        }
        data.setClassIndex(data.numAttributes() - 1);
    }


    /*
     * Note: in normal use, numInstances will be -1, because that's not the place where
     * the choice of numInstances is actually made. But if for some reason we ask for fewer, return fewer.
     */
    @Override
    public Instances generateTrainingPositives(int numInstances) {
        return getEnoughInstances(trainingSample, numInstances);
    }

    public Instances generateTestingData(int numInstances) {
        return getEnoughInstances(allData, numInstances);
    }

    @Override
    public Instances generateTrainingNegatives(int numInstances) {
        return null;
    }

    @Override
    public Instances generateTestingPositives(int numInstances) {
        return null;
    }

    @Override
    public Instances generateTestingNegatives(int numInstances) {
        return null;
    }

}
