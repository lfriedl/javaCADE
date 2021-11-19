package cade.generativeProbDensities;

import cade.Parameters;
import cade.positiveDataGenerators.DataGenerator;
import weka.core.AttributeStats;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import cade.estimators.Estimator;
import cade.estimators.UniformNominalEstimator;
import cade.estimators.UniformNumericEstimator;


public class UniformProbDensity extends IndepMarginalsProbDensity {
    public double[] min, max;
    private boolean[] isInteger; //if it's numeric, true if this attribute has only integer values
    int numAttributes;
    
    public UniformProbDensity(DataGenerator DG){
    	super(DG);
        calculateAttributeProperties(Parameters.UniformPNegRange.POS_RANGE, true);
    }
    
    public UniformProbDensity(Parameters params) {
    	super(params);
        
//    	Instances trainingInsts;
//    	if (dataGen.trainingSample != null)
//    		trainingInsts = dataGen.trainingSample;
//    	else{
//    		trainingInsts = params.getDataGenerator().generateTrainingPositives(dataGen.numTrainingPositives);
//		}

        calculateAttributeProperties(params.uniformPNegRange, params.useIntegerUniformPNegs);
    }

/*

    // With this version of the constructor, we won't ever call calculateAttributeProperties() and constructEstimators().
    public UniformProbDensity(Estimator[] estimators){
        super();
        estimatorsForAttributes = estimators;
    }
*/

    // Sets object members: min, max, isNumeric, etc.
    protected void calculateAttributeProperties(Parameters.UniformPNegRange uniformPNegRange,
                                                boolean useIntegerUniformPNegs) {
        // dataGen.trainingSample should already have been created
        numAttributes = dataGen.trainingSample.numAttributes() - 1;

        min = new double[numAttributes];
        max = new double[numAttributes];
        isInteger = new boolean[numAttributes];

        for (int i = 0; i < numAttributes; i++) {
            // it's only meaningful to calculate min & max for the numeric attributes

//            isNumeric[i] = dataGen.trainingSample.attribute(i).isNumeric();
            if (dataGen.trainingSample.attribute(i).isNumeric()) {
                min[i] = calcPseudoNegMin(i, uniformPNegRange);
                max[i] = calcPseudoNegMax(i, uniformPNegRange);

                if (useIntegerUniformPNegs){
                    AttributeStats as = dataGen.trainingSample.attributeStats(i);
                    if (as.realCount == 0)
                        isInteger[i] = true;
                }
            }
        }
    }

    //Calculates the min for uniform pseudo negatives, based on the method specified in Parameters and the properties in the data generator
	public double calcPseudoNegMin(int attr, Parameters.UniformPNegRange uniformPNegRange){
		if (uniformPNegRange == Parameters.UniformPNegRange.POS_RANGE)
			return dataGen.min[attr];
		else if (uniformPNegRange == Parameters.UniformPNegRange.FIVE_STDDEVS_FROM_MEAN)
			return dataGen.posMeans[attr] - 5 * dataGen.posStdDevs[attr];
		else
			return dataGen.min[attr] - 2 * dataGen.posStdDevs[attr];
	}

	//Calculates the max for uniform pseudo negatives, based on the method specified in cade.Parameters and the properties in the data generator
	public double calcPseudoNegMax(int attr, Parameters.UniformPNegRange uniformPNegRange){
		if (uniformPNegRange == Parameters.UniformPNegRange.POS_RANGE)
			return dataGen.max[attr];
		else if (uniformPNegRange == Parameters.UniformPNegRange.FIVE_STDDEVS_FROM_MEAN)
			return dataGen.posMeans[attr] + 5 * dataGen.posStdDevs[attr];
		else
			return dataGen.max[attr] + 2 * dataGen.posStdDevs[attr];
	}

    public Estimator[] constructEstimators() {
        estimatorsForAttributes = new Estimator[numAttributes];
        for (int i = 0; i < numAttributes; i++) {
            if (!dataGen.trainingSample.attribute(i).isNumeric())  // it's categorical (nominal)
                estimatorsForAttributes[i] = new UniformNominalEstimator(dataGen.valuesSeen[i]);
            else                // for density estimation, isInteger[i] doesn't make a difference
                estimatorsForAttributes[i] = new UniformNumericEstimator(min[i], max[i]);
        }
        return estimatorsForAttributes;
    }

    @Override
    public Instances generateItems(int numInstances) throws Exception {
        if (numInstances == -1)
            numInstances = dataGen.numTrainingPositives();//50:50 positive:pNeg ratio

        Instances instances = generateUniformPseudoNegatives(numInstances);
        setClassLabelNegative(instances);
        return instances;
    }

    protected Instances generateUniformPseudoNegatives(int numInstances){

            Instances instances = new Instances(dataGen.trainingSample, numInstances);//creates an empty data set using header info

            for (int currInst = 0; currInst < numInstances; currInst++){ //create numInstances instances
                Instance dataPoint = new DenseInstance(dataGen.trainingSample.instance(0));
                dataPoint.setDataset(instances);
                for (int i=0; i<numAttributes; i++){	//loop over all the attributes
                    if (dataGen.trainingSample.attribute(i).isNumeric()) {
                        //get a random # between the min and max for this attribute
                        double pseudoNegMin = min[i];
                        double pseudoNegMax = max[i];
                        double val;
                        if (isInteger[i])
                        	val = rng.nextInt((int)(pseudoNegMax-pseudoNegMin)+1) + pseudoNegMin;
                        else
                        	val = rng.nextDouble()*(pseudoNegMax-pseudoNegMin) + pseudoNegMin;

                        dataPoint.setValue(i, val);
                    } else {
                        int valIndex = rng.nextInt(dataGen.trainingSample.attribute(i).numValues());
                        dataPoint.setValue(i, dataGen.trainingSample.attribute(i).value(valIndex));
                    }
                }
                instances.add(dataPoint);
            }

            return instances;
        }

    public String getName(){
		return("Uniform");
	}
}
