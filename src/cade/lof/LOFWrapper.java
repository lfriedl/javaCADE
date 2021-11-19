package cade.lof;

import cade.Evaluator;
import weka.classifiers.evaluation.NominalPrediction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Random;

/**
 * Class to call multiple LOF variants and evaluate using the same train/test data for each model:
 * -plain LOF
 * -bagged LOF (default bagging settings), with two combiner methods: sumLOF and BreadthFirstLOF
 */
public class LOFWrapper {
    protected LOFRunner singleLOF;
    protected LOFRunner baggedLOF;
    protected Random randomness;    // for repeatable mode
    protected int numBaggingTrials = 10;
    public int numMethodsIncluded = 2;  // = how many aucs we'll return. Internals compute several scores, but only 2 are any good.

    static boolean printedInfoAlready = false;

    public LOFWrapper(Random randomness) {
        singleLOF = new LOFRunner();
        baggedLOF = new LOFRunner();
        this.randomness = randomness;
        if (!printedInfoAlready) {
            System.out.println("LOF: " + numMethodsIncluded + " methods, and bagging uses t = " + numBaggingTrials);
            printedInfoAlready = true;
        }
    }

    public void buildModels(Instances trainingData) throws Exception {
        singleLOF.addLOFAttributeToSingleDataset(trainingData);
        baggedLOF.runBaggedLOFSingleDataset(trainingData, numBaggingTrials, randomness);
    }

    // The AUC scores returned are:
    // { plainLOF, Bagged sumLOF, Bagged BreadthFirstLOF }
    public double[] getAUCsForNewTestSet(Instances testData) throws Exception {
        double[] aucArray = new double[numMethodsIncluded];

        Instances testDataWithLOF = singleLOF.addLOFAttributeUsingExistingModel(testData);
        aucArray[0] = getAUCOfAttribute(testDataWithLOF, testDataWithLOF.numAttributes() - 1);

        Instances testDataWith2LOFScores = baggedLOF.addBaggedLOFAttributeUsingExistingModel(testData);

        aucArray[1] = getAUCOfAttribute(testDataWith2LOFScores, testDataWith2LOFScores.numAttributes() - 3);
        // skipped "MultiplyLikeDensities", because it's always terrible
        // skipping "BreadthFirstLOF" as well
        //aucArray[2] = getAUCOfAttribute(testDataWith2LOFScores, testDataWith2LOFScores.numAttributes() - 1);

        return aucArray;
    }

    public ArrayList<NominalPrediction>[] getPredsForNewTestSet(Instances testData) throws Exception {
    	ArrayList<NominalPrediction>[] predArray = new ArrayList[numMethodsIncluded];
    	
    	Instances testDataWithLOF = singleLOF.addLOFAttributeUsingExistingModel(testData);
    	Instances testDataWith2LOFScores = baggedLOF.addBaggedLOFAttributeUsingExistingModel(testData);
    	
    	predArray[0] = getPredOfAttribute(testDataWithLOF, testDataWithLOF.numAttributes() - 1);
    	predArray[1] = getPredOfAttribute(testDataWith2LOFScores, testDataWith2LOFScores.numAttributes() - 3);
    	
    	return predArray;
    }
    
    // Assumes higher = more outlying
    protected double getAUCOfAttribute(Instances data, int attrIndex) throws Exception {
        return Evaluator.getAUCFromWeka(getPredOfAttribute(data, attrIndex));
    }

    //prediction version of above method
 // Assumes higher = more outlying
    protected ArrayList<NominalPrediction> getPredOfAttribute(Instances data, int attrIndex) throws Exception {
        ArrayList<NominalPrediction> predArray = new ArrayList<NominalPrediction>();
        for (int i = 0; i < data.numInstances(); i++) {
            Instance inst = data.instance(i);
            double lofVal = inst.value(attrIndex);
            NominalPrediction nomPred = new NominalPrediction(inst.classValue(),
                    new double[]{-1 * lofVal, lofVal});
            predArray.add(nomPred);
        }
        return predArray;
    }

}
