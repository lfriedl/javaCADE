package cade.generativeProbDensities;

import cade.Parameters;
import cade.estimators.Estimator;
import cade.estimators.MixtureEstimator;
import weka.core.Instance;
import weka.core.Instances;

public class MixurePNegGenerator extends ProbDensity {
    protected GenerativeProbDensity unifPseudoNegGenerator;
    protected GenerativeProbDensity margPseudoNegGenerator;
    protected double fractionUniform;

    public MixurePNegGenerator(Parameters params, double fractionUniform) {
        super(params);
        unifPseudoNegGenerator = new UniformProbDensity(params);
        margPseudoNegGenerator = new IndepMarginalsProbDensity(params);
        this.fractionUniform = fractionUniform;
    }

    public Instances generateItems(int numInstances) throws Exception {
        if (numInstances == -1)
            numInstances = dataGen.numTrainingPositives();

        // flip a coin to determine how many to generate from each class
        int numUnifWanted = 0;
        for (int i = 0; i < numInstances; i++) {
            double flip = rng.nextDouble();
            if (flip <= fractionUniform)
                    numUnifWanted++;
        }

        Instances uniforms = unifPseudoNegGenerator.generateItems(numUnifWanted);
        Instances marginals = margPseudoNegGenerator.generateItems(numInstances - numUnifWanted);

        for (int i = 0; i < uniforms.numInstances(); i++) {
            Instance unifInstance = uniforms.instance(i);
            marginals.add(unifInstance);
        }

        return marginals;
    }


    public Estimator[] constructEstimators() {
        Estimator[] estimatorsToReturn = new Estimator[dataGen.numAttributes()];
        Estimator[] unifEstimators = ((ProbDensity)unifPseudoNegGenerator).getEstimatorsForAttributes();
        Estimator[] margEstimators = ((ProbDensity)margPseudoNegGenerator).getEstimatorsForAttributes();

        for (int i = 0; i < estimatorsToReturn.length; i++) {
            estimatorsToReturn[i] = new MixtureEstimator(unifEstimators[i], margEstimators[i], fractionUniform);
        }

        return estimatorsToReturn;
    }
    
    public String getName(){
		return("Mixture");
	}
}
