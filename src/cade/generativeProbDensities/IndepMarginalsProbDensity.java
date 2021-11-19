package cade.generativeProbDensities;

import cade.Parameters;
import cade.positiveDataGenerators.DataGenerator;
import weka.core.AttributeStats;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import cade.estimators.CategoricalEstimator;
import cade.estimators.Estimator;
import cade.estimators.KDEstimator;
import cade.estimators.UniformNumericEstimator;
import java.io.File;


public class IndepMarginalsProbDensity extends ProbDensity {

    public IndepMarginalsProbDensity(Parameters params) {
        super(params);
   }
    
    public IndepMarginalsProbDensity(DataGenerator dataGen){
    	super(dataGen);
    }

    // For arbitrary observed data, we'll use KDE and Categorical estimators
    public Estimator[] constructEstimators() {
        Estimator[] estimators = new Estimator[dataGen.numAttributes()];
        // We need the training positives back; they are
        // the distribution we generated our pNegs from, and what are modeling
        
        Instances trainingPositives;
        if (dataGen.trainingSample != null)
        	trainingPositives = dataGen.trainingSample;
        else{
        	trainingPositives = dataGen.generateTrainingPositives(dataGen.numTrainingPositives);
        }
        // note: training data has a class attribute too, but it's always last, so this loop won't get to it
        for (int i = 0; i < dataGen.numAttributes(); i++) {
            if (trainingPositives.attribute(i).isNominal())
                estimators[i] = new CategoricalEstimator(trainingPositives, i);
            else {
                AttributeStats at = trainingPositives.attributeStats(i);
                // If there's only 1 value seen, KDE would give an error.
                if (at.distinctCount > 1) {
                    estimators[i] = new KDEstimator(trainingPositives, i);
                } else {
                    estimators[i] = new UniformNumericEstimator(at.numericStats.min, at.numericStats.min);
                }
            }
        }
        return estimators;
    }

    //numInstances = -1 means to use the default setting (same number of pseudo negatives as there are positives);
    // otherwise, just make numInstances pseudoNegs
    public Instances generateItems(int numInstances) throws Exception {
        if (numInstances == -1)
            numInstances = dataGen.numTrainingPositives();

        Instances insts = samplePseudoNegativesFromIndepFeatures(numInstances);

        insts.setClassIndex(insts.numAttributes() - 1);
        setClassLabelNegative(insts);
        
        return insts;
    }

    /*
     * We generate pNegs simply by sampling from the empirical distribution of each feature.
     */
    protected Instances samplePseudoNegativesFromIndepFeatures(int numInstances) {
        //set up things that don't depend on the pNeg generation type
        int numAttributesToSample = dataGen.numAttributes();    // skip the class variable

        Instances trainingData;
        if (dataGen.trainingSample != null)
        	trainingData = dataGen.trainingSample;
        else
        	trainingData = dataGen.generateTrainingPositives(dataGen.numTrainingPositives);  // which we'll permute to generate our new samples

//        Instance exampleInstance = trainingData.firstInstance();  // Could also use dataGen.getOneExampleInstance(),
                                                                  // but it would affect the Random() and we'd need to change the tests.
        Instances newInstances = new Instances(trainingData, numInstances);//creates an empty data set using header info from data

        for (int currInst = 0; currInst < numInstances; currInst++) { //create numInstances instances
            Instance newDataPoint = new DenseInstance(trainingData.firstInstance().numAttributes());
            newDataPoint.setDataset(newInstances);
            for (int i = 0; i < numAttributesToSample; i++) {    //sample each attribute from the training data
                int index = rng.nextInt(trainingData.numInstances());
                newDataPoint.setValue(i, trainingData.instance(index).value(i));
            }
            newInstances.add(newDataPoint);
        }

        return newInstances;
    }


    public String getName(){
		return("KDE");
	}

    public void doneWithProbDensity() {
        for (Estimator est : getEstimatorsForAttributes()) {
            if (est instanceof KDEstimator) {
                new File(((KDEstimator) est).fileName).delete();
            }
        }
    }
}
