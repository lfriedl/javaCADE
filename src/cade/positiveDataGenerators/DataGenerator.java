package cade.positiveDataGenerators;

import java.io.IOException;
import java.util.HashSet;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;

public abstract class DataGenerator {
	public double[] posMeans, posStdDevs, min, max;
	public HashSet<Integer>[] valuesSeen;
	public double percentPositive;
	public int numAttributes;
    public int numTrainingPositives; // Used in deciding how many pNegs to generate

	public Instances trainingSample;	// Moved up to this class because it's needed even for synthetic data.
										// (Previously there was a separate but equivalent field here, named mostRecentTrainingPositives.)

    protected Random randomness;
    public Random getRandomGenerator() {
        return randomness;
    }

    public abstract void loadData() throws IOException;
	public abstract void calculatePropertiesFromData();  // Sets the above properties

    //numInstances = -1 means use all the training/testing data available (only meaningful for non-synthetic data)
	public abstract Instances generateTrainingPositives(int numInstances);
	public abstract Instances generateTrainingNegatives(int numInstances);//used for a baseline
	public abstract Instances generateTestingPositives(int numInstances);
	public abstract Instances generateTestingNegatives(int numInstances);

    // Not counting the class attribute
	public int numAttributes() {
        return numAttributes;
    }

    // how many positives were provided during training?
	public int numTrainingPositives() {
        return numTrainingPositives;
    }

    public void prepareForNextRun() {}  // defaults to doing nothing

    // Utility function.
    public Instance getOneExampleInstance() {
        return generateTrainingPositives(1).firstInstance();
    }

	public Instances getEnoughInstances(Instances insts, int numInstances){
	    if (numInstances >= insts.numInstances() || numInstances < 0)
	    	return new Instances(insts);
	    
     	boolean[] dataPointsUsed = new boolean[insts.numInstances()];
	        for (int i=0; i<insts.numInstances(); i++)
	        	dataPointsUsed[i] = false;

        if (randomness == null)
	        randomness = new Random();
	        
     	int instanceCount = 0;
     	
        Instances instances = new Instances(insts, numInstances);//constructor that just copies header info of insts, but none of the instances
        while (instanceCount < numInstances) {
        	int i = randomness.nextInt(insts.numInstances());
        	if (!dataPointsUsed[i]){
        		instances.add(insts.instance(i));
        		dataPointsUsed[i] = true;
        		instanceCount++;
        	}
        }
        return instances;
	}

}
