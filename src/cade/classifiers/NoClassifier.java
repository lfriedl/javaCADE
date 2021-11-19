package cade.classifiers;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class NoClassifier extends LocalClassifier {

	//wekaClassifier = null, passed in from cade.classifiers.MultipleClassifierFDC
	public NoClassifier(Classifier wekaClassifier) {
		super(wekaClassifier);
	}

    public String toString() { return "NOCLASSIFIER"; }

	//no classifier to build!
	public void buildClassifier() {}
    public boolean addTrainingData(Instances newData) { return true; }

	
    public double[] makePredictions(Instances testInstances) {
    	double[] preds = new double[testInstances.numInstances()];
        for (int i = 0; i < testInstances.numInstances(); i++)
            preds[i] = 0.5;
        return preds;
    }
    
    public double[] getDistributionForInstance(Instance inst) {
        return new double[]{.5, .5};
    }
}
