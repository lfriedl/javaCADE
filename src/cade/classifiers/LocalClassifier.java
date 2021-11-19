package cade.classifiers;

import java.util.ArrayList;

import cade.Evaluator;
import cade.Parameters;
import cade.StatsZeroCounter;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.functions.Logistic;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
//import weka.classifiers.trees.SimpleCart;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;


public class LocalClassifier {

    public weka.classifiers.Classifier wekaClassifier;
    protected Instances trainingData = null;

    public LocalClassifier(Parameters.ClassifierType classifierType){
        try {
            initialize(classifierType);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void initialize(Parameters.ClassifierType classifierType) throws Exception {
    	switch (classifierType) {
        case NAIVEBAYES:
            wekaClassifier = new NaiveBayes();
            ((NaiveBayes) wekaClassifier).setUseKernelEstimator(true);
            break;
        case TREE:
//        	if (pNegType == Parameters.PseudoNegGenerationType.MARGINAL_KDE){
//            	wekaClassifier = new SimpleCart();
//                ((SimpleCart)wekaClassifier).setUseLaplace(true);
//                ((SimpleCart)wekaClassifier).setUsePrune(false);
//                ((SimpleCart) wekaClassifier).setUseOneSE(true);
//                System.out.println("SimpleCart!");
//        	} else{
                wekaClassifier = new J48();
//            ((J48)wekaClassifier).setOptions(new String[]{"-U", "-A", "-O"});
                ((J48)wekaClassifier).setUnpruned(true);
                ((J48)wekaClassifier).setUseLaplace(true);
//                ((J48)wekaClassifier).setCollapse(false);   // under weka-3-6
                ((J48)wekaClassifier).setCollapseTree(false);   // under weka-3-8
                System.out.println("J48!");
//        	}
            break;
        case KNN:
            wekaClassifier = new IBk(200);    // use this normally
            ((IBk)wekaClassifier).setDistanceWeighting(new SelectedTag(IBk.WEIGHT_INVERSE, IBk.TAGS_WEIGHTING));
            break;
        case LOGISTIC:
		    wekaClassifier = new Logistic();
            break;
        case RANDOMFOREST:
            wekaClassifier = new RandomForest();
            int numTrees = 200;
//            ((RandomForest) wekaClassifier).setNumTrees(numTrees);
//            ((RandomForest) wekaClassifier).setNumFoldsForBackfitting(2);	// or 0 for none
            ((RandomForest) wekaClassifier).setOptions(new String[]{"-I", String.valueOf(numTrees), "-N", "2",
                                                                    "-num-slots", "0"});

            break;

        default:
            try {
                throw (new Exception("Classifier type of " + classifierType + " not supported"));
            } catch (Exception e) {
                e.printStackTrace();
            }
    }
    }
    // Option to construct the weka classifier elsewhere and pass it in.
    public LocalClassifier(weka.classifiers.Classifier wekaClassifier) {
        this.wekaClassifier = wekaClassifier;
    }

    // Adds newData to trainingData. Can be called multiple times.
    // newData must have its classIndex already set.
    public boolean addTrainingData(Instances newData) {

        if (trainingData == null) {
            trainingData = new Instances(newData);

        } else {

            // Check that the new instances are compatible
            Instance firstInst = newData.firstInstance();
            boolean isCompatible = trainingData.checkInstance(firstInst);
            if (!isCompatible) {
                return false;
            }

            // Add them in
            for (int i = 0; i < newData.numInstances(); i++) {
                trainingData.add(newData.instance(i));
            }
        }

        return true;
    }

    public void buildClassifier() throws Exception {
        wekaClassifier.buildClassifier(trainingData);
    }

    public double[] getAUCForTestSet(Instances testInstances) throws Exception {
        ArrayList<NominalPrediction> predArray = getPredictionsArrayForTestSet(testInstances);

        // count ties in these too
        boolean countTies = true;
        if (countTies) {
            double[] scoresAfter = new double[predArray.size()];
            for (int i = 0; i < predArray.size(); i++) {
                scoresAfter[i] = predArray.get(i).distribution()[0];
            }
            double[] tiesAfter = StatsZeroCounter.findLongestTieValues(scoresAfter);
            System.out.println("\tclassifier tied values: " + (int) tiesAfter[0] + " having P(+)=" + tiesAfter[1]);
        }

//        return new double[] { (new Evaluator()).getAUCFromPredArrayUsingPerl(predArray) };
        return new double[] { Evaluator.getAUCFromWeka(predArray) };
    }
    
    // Next four methods use wekaClassifier.distributionForInstance(), varying in return type.
    public ArrayList<NominalPrediction> getPredictionsArrayForTestSet(Instances testInstances) throws Exception {
    	ArrayList<NominalPrediction> predArray = new ArrayList<NominalPrediction>();
    	for (int inst = 0; inst < testInstances.numInstances(); inst++) {
    		predArray.add(new NominalPrediction(
                    testInstances.instance(inst).classValue(), getDistributionForInstance(testInstances.instance(inst))));
    	}
    	return predArray;
    }

    public double[][] getDistributionsForTestSet(Instances testInstances) throws Exception {
        double[][] distArray = new double[testInstances.numInstances()][];
        for (int i = 0; i < testInstances.numInstances(); i++) {
            double[] distributionForInstance = getDistributionForInstance(testInstances.instance(i));
            distArray[i] = distributionForInstance;
        }
        return distArray;
    }

    public double[] getDistributionForInstance(Instance inst) throws Exception {
        return (wekaClassifier.distributionForInstance(inst));
    }

    // Like getDistributions, but only stores the P(+) element
    public double[] makePredictions(Instances testInstances) throws Exception {
        double[] preds = new double[testInstances.numInstances()];
        for (int i = 0; i < testInstances.numInstances(); i++) {
            double[] probs = getDistributionForInstance(testInstances.instance(i));
            preds[i] = probs[0];
        }
        return preds;
    }

    public String toString(){
    	return "" + wekaClassifier.getClass();
    }
}
