package cade.generativeProbDensities;

import java.util.Random;

import cade.positiveDataGenerators.DataGenerator;
import weka.core.Instances;

// Interface implemented by ProbDensity (a density estimator) and by FormulaDensityCombiner (which combines a
// ProbDensity with a Classifier)

public interface GenerativeProbDensity {
	DataGenerator getDataGenerator();
	Instances generateItems(int numInstances) throws Exception;
	double[] computeLogDensity(Instances testInstances);

	void setRandom(Random r);
    String getName();

	default void doneWithProbDensity() {}
}
