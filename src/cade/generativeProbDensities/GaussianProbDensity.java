package cade.generativeProbDensities;

import cade.Parameters;
import cade.estimators.CategoricalEstimator;
import cade.estimators.Estimator;
import cade.estimators.GaussianEstimator;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;


public class GaussianProbDensity extends ProbDensity {

	public GaussianProbDensity(Parameters params) {
		super(params);
	}

	@Override
	public Estimator[] constructEstimators() {
		Estimator[] estimators = new Estimator[dataGen.numAttributes()];

		Instances trainingPositives;
		if (dataGen.trainingSample != null)
			trainingPositives = dataGen.trainingSample;
		else{
			trainingPositives = dataGen.generateTrainingPositives(dataGen.numTrainingPositives);
		}

		for (int i = 0; i < dataGen.numAttributes(); i++) {
			if (trainingPositives.attribute(i).isNominal())
				estimators[i] = new CategoricalEstimator(trainingPositives, i);
			else {
				double mean = trainingPositives.meanOrMode(i);
				double stdDev = Math.sqrt(trainingPositives.variance(i));
				estimators[i] = new GaussianEstimator(mean, stdDev, rng);
			}
		}
		return estimators;
	}

	@Override
	public Instances generateItems(int numInstances) throws Exception {
		constructEstimators();
        
        //now generate pNegs
		if (numInstances == -1)
			numInstances = dataGen.numTrainingPositives();

		Instance exampleInstance = dataGen.getOneExampleInstance();
        Instances instances = new Instances(dataGen.trainingSample, numInstances);//creates an empty data set using header info from data

        for (int currInst = 0; currInst < numInstances; currInst++){ //create numInstances instances
        	Instance dataPoint = new DenseInstance(exampleInstance);
        	dataPoint.setDataset(instances);
        	for (int i=0; i<getEstimatorsForAttributes().length; i++){	//loop over all the attributes
        		if (!exampleInstance.attribute(i).isNominal()) {
        			double val = ((GaussianEstimator) estimatorsForAttributes[i]).generate();
        			dataPoint.setValue(i, val);
        		} else {  // If categorical, sample directly from the training data
        			int valIndex = rng.nextInt(dataGen.trainingSample.attribute(i).numValues());
        			dataPoint.setValue(i, dataGen.trainingSample.attribute(i).value(valIndex));
        		}
        	}
        	instances.add(dataPoint);
        }
        
        //Now label them negative
        instances.setClassIndex(instances.numAttributes() - 1);
        setClassLabelNegative(instances);
        
        return instances;
	}

	public String getName(){
		return("Gaussian");
	}
}
