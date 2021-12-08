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

    public int[] idAttributes;  //indexed from 1
    public Instances savedIDAttrs;

    public NoLabelsArffDataGenerator(String dataFile, int[] attributesToUse, int[] idAttributes,
                                     int numTrainPositives, Random rng, boolean ignoreInstancesWithMissingValues) throws IOException {
        this.dataFile = dataFile;
        this.attributesToUse = attributesToUse; // may have length 0, meaning "use all except IDattrs"
        this.idAttributes = idAttributes;   // may be [-1], meaning "there are none"
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

        // Create new class label.
        // (If data already contained a class label, include it among idAttributes in order not to use it as a feature.)
        setUpClassLabel(allData);       // makes them all "positive"
        percentPositive = 100;

        if (idAttributes[0] != -1) {
            storeIDsForLater();
            buildAttrsToUse();  // constructs the attributesToUse array if it was passed in empty
        }
        removeUnwantedAttributes(allData, attributesToUse);

        //numAttributes doesn't include class label
        numAttributes = allData.numAttributes() - 1;

        System.out.println("Using " + numAttributes + " attributes");
        System.out.println("Preparing training sample");

        // When running in unsupervised mode, we calculate data ranges using entire data set (even if we use only a
        // subset for training).
        // (Reasoning is: in this setting we really do know the testing instances in advance, so it's ok to use that info.
        // In contrast, in supervised mode we calculate them based only on the training set, because (for example) we
        // can't assume the negative class would be rare.)

        // (code for this is a bit kludgy.)
        trainingSample = allData; // trainingSample is what calculatePropertiesFromData() runs on.
        int storeNumTrainingPositives = numTrainingPositives; // saved because calculatePropertiesFromData() will clobber it
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
            boolean isIDAttr = false;
            for (int idAttrNum : idAttributes) {
                if (idAttrNum == attrNum1Based)
                    isIDAttr = true;
            }
            if (!isIDAttr) {
                attributesToUse[indexInNewAttrsToUse] = attrNum1Based;
                indexInNewAttrsToUse++;
            }
        }
    }

    private void storeIDsForLater() {

        // save them for later in a new Instances object
        savedIDAttrs = new Instances(allData);    // makes a copy
        removeUnwantedAttributes(savedIDAttrs, idAttributes);   // removes all but the idAttrs and classIndex
        if (savedIDAttrs.classIndex() >= 0) {
            int classIndexToRm = savedIDAttrs.classIndex();
            savedIDAttrs.setClassIndex(-1); // unset class index so that weka lets us remove this attr
            savedIDAttrs.deleteAttributeAt(classIndexToRm);
        }

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

    // These aren't meaningful for unlabeled data.
    @Override
    public Instances generateTestingPositives(int numInstances) {
        return null;
    }

    @Override
    public Instances generateTrainingNegatives(int numInstances) {
        return null;
    }

    @Override
    public Instances generateTestingNegatives(int numInstances) {
        return null;
    }

}
