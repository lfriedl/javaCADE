package cade.lof;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.LOF;

import java.util.Random;

/**
 * Class assumes throughout that class label is the last attribute of input data. Adds one or more LOF attrs after it.
 */
class LOFRunner {
    public static int numThreads = 25;
    public static int minK = 10;    // minK and maxK are initialized to the defaults used in LOF.java
    public static int maxK = 40;

    protected LOF singleTrainedLOF;
    protected LOF[] trainedLOFs;
    protected int[][] attrSubsets;
    protected int[] attrSubsetCutoffs;


    public Instances addLOFAttributeToSingleDataset(Instances allData) throws Exception {

        LOF lof = new LOF();
        String[] options = new String[]{"-num-slots", String.valueOf(numThreads),
                "-min", String.valueOf(minK), "-max", String.valueOf(maxK)};
        lof.setOptions(options);
        lof.setInputFormat(allData);
        Instances newData = Filter.useFilter(allData, lof);

        // Store this for possible re-use
        singleTrainedLOF = lof;

        return newData;
    }

    public Instances addLOFAttributeUsingExistingModel(Instances testData) throws Exception {
        if (singleTrainedLOF == null) {
            throw new Exception("Must train addLOFAttributeToSingleDataset before calling this");
        }
        return Filter.useFilter(testData, singleTrainedLOF);
    }

    /**
     * Implemented like Lazarevic & Kumar:
     *  -Always 10 rounds of bagging (as per their Fig 11)
     *  -Each bag: d/2 to (d-1) features (choose number uniformly)
     *  -Choose features uniformly w/o replacement
     *  -Combining: two methods. We'll do both and return one attribute with each score: sumLOF, then BreadthFirstLOF
     *
     *  Assumes class label is the last attribute of data.
     */
    public Instances runBaggedLOFSingleDataset(Instances data, int numTrials, Random randomness) throws Exception {
        int numNonClassAttrs = data.numAttributes() - 1;
        int dOver2 = (int) Math.ceil(numNonClassAttrs / 2.0);
        return runBaggedLOFSingleDataset(data, numTrials, dOver2, numNonClassAttrs - 1, randomness);
    }

    // randomness is allowed to be null
    public Instances runBaggedLOFSingleDataset(Instances data, int numTrials, int minNumAttrs, int maxNumAttrs, Random randomness) throws Exception {

        trainedLOFs = new LOF[numTrials];
        attrSubsets = new int[numTrials][];
        attrSubsetCutoffs = new int[numTrials];

        int numNonClassAttrs = data.numAttributes() - 1;
        if (maxNumAttrs >= numNonClassAttrs) {
            System.err.println("in runBaggedLOF, decreasing maxNumAttrs from impossible value");
            maxNumAttrs = numNonClassAttrs - 1;
        }

        Instances[] LOFRuns = new Instances[numTrials];

        if (randomness == null)
            randomness = new Random();

        for (int t = 0; t < numTrials; t++) {
            // Choose random number of attributes
            int numAttrsToUse = minNumAttrs + randomness.nextInt(maxNumAttrs + 1 - minNumAttrs);

            // Choose attributes
            // in this array, 0 to (numAttrsToUse - 1) are the (0-based indices of) ones to keep
            int[] attrIndicesPlus = sampleWithoutReplacement(numAttrsToUse, numNonClassAttrs, randomness);

            // Prepare an Instances having just those attributes plus the class label
            Instances instancesWithAttrSubset = subsetUsingTheseAttributes(data, attrIndicesPlus, numAttrsToUse);

            // Run LOF
            LOFRuns[t] = addLOFAttributeToSingleDataset(instancesWithAttrSubset);

            // Store for future use
            attrSubsets[t] = attrIndicesPlus;
            attrSubsetCutoffs[t] = numAttrsToUse;
            trainedLOFs[t] = singleTrainedLOF;

        }

        // Now, combine the scores. Each of these calls adds an attribute.
        data = combineScoresCumulativeSum(data, LOFRuns);
        data = combineScoresMultiplyLikeDensities(data, LOFRuns);
        data = combineScoresBreadthFirst(data, LOFRuns);    // This call also adds an attribute to LOFRuns

        return data;
    }

    public Instances addBaggedLOFAttributeUsingExistingModel(Instances testData) throws Exception {
        if (trainedLOFs.length == 0 ||
                (trainedLOFs.length != attrSubsets.length) || (trainedLOFs.length != attrSubsetCutoffs.length)) {
            throw new Exception("Bagged LOF models don't seem to have been trained");
        }

        int numTrials = trainedLOFs.length;
        Instances[] LOFRuns = new Instances[numTrials];

        for (int t = 0; t < numTrials; t++) {

            // take appropriate subset of attributes
            Instances instancesWithAttrSubset = subsetUsingTheseAttributes(testData, attrSubsets[t], attrSubsetCutoffs[t]);

            // run LOF
            singleTrainedLOF = trainedLOFs[t];
            LOFRuns[t] = addLOFAttributeUsingExistingModel(instancesWithAttrSubset);
        }

        // Now, combine the scores. Each of these calls adds an attribute.
        testData = combineScoresCumulativeSum(testData, LOFRuns);
        testData = combineScoresMultiplyLikeDensities(testData, LOFRuns);
        testData = combineScoresBreadthFirst(testData, LOFRuns);    // This call also adds an attribute to LOFRuns

        return testData;
    }

    protected Instances subsetUsingTheseAttributes(Instances data, int[] attrSubset, int numAttrsToUse) {

            // Prepare an Instances having just those attributes plus the class label
            Instances instancesWithAttrSubset = new Instances(data);

            // delete the attributes we're not keeping.
            // How to delete them in a consistent order? Let's do the following (though inefficient):
            // for attrNum = numNonClassAttrs to 1:
            //      see if it's in the "keep" or "delete" list.
            //      delete if you're supposed to.

            for (int i = attrSubset.length - 1; i >= 0; i--) {    // 0-based attr num
                boolean deleteMe = false;
                for (int j = numAttrsToUse; j < attrSubset.length; j++) {  // 0-based index into attrIndicesPlus
                    if (attrSubset[j] == i) {
                        deleteMe = true;
                        break;
                    }
                }
                if (deleteMe) {
                    instancesWithAttrSubset.deleteAttributeAt(i);
                }

            }
        return instancesWithAttrSubset;
    }


    /**
     * For each Instances in lofRuns, take its LOF attribute (in final position).
     * Combine them by summing them all.
     * Append the new score as an attribute to data
     * @param data
     * @param lofRuns
     * @return
     */
    private Instances combineScoresCumulativeSum(Instances data, Instances[] lofRuns) {

        Instances newData = new Instances(data);

        Attribute sumLOFAttr = new Attribute("sumLOF");
         newData.insertAttributeAt(sumLOFAttr, newData.numAttributes());
        for (int i = 0; i < newData.numInstances(); i++) {
            Instance inst = newData.instance(i);

            double sumLOF = 0;
            for (int t = 0; t < lofRuns.length; t++) {
                sumLOF += lofRuns[t].instance(i).value(lofRuns[t].numAttributes() - 1);
//                if (! (lofRuns[t].instance(i).value(lofRuns[t].numAttributes() - 1) < 1000)) {
//                    System.out.println("found some bad values in t = " + t + ", i = " + i);
//                }
            }
            inst.setValue(newData.numAttributes() - 1, sumLOF);
        }
        return newData;
    }

    private Instances combineScoresMultiplyLikeDensities(Instances data, Instances[] lofRuns) {
        Instances newData = new Instances(data);

        Attribute prodLOFAttr = new Attribute("productLOF");
         newData.insertAttributeAt(prodLOFAttr, newData.numAttributes());
        for (int i = 0; i < newData.numInstances(); i++) {
            Instance inst = newData.instance(i);

            double prodLOF = 1;
            for (int t = 0; t < lofRuns.length; t++) {
                prodLOF *= 1 / lofRuns[t].instance(i).value(lofRuns[t].numAttributes() - 1);
//                }
            }
            inst.setValue(newData.numAttributes() - 1, 1/prodLOF);
        }
        return newData;
    }

    /**
     * For each Instances in lofRuns, take its LOF attribute.
     * Combine them using breadth first search. (Taking first element in one ranking, first in the next, etc.)
     * Append the new score as an attribute to data.
     * Also adds an attribute to LOFRuns.
     * @param data
     * @param lofRuns
     * @return
     */
    private Instances combineScoresBreadthFirst(Instances data, Instances[] lofRuns) {

        // Number the items 0 to data.numInstances()-1 so that we can recognize them later
        Attribute itemIndexAttr = new Attribute("itemIndex");
        for (int t = 0; t < lofRuns.length; t++) {
            lofRuns[t].insertAttributeAt(itemIndexAttr, lofRuns[t].numAttributes());
            int counter = 0;
            for (int i = 0; i < data.numInstances(); i++) {
                lofRuns[t].instance(i).setValue(lofRuns[t].numAttributes() - 1, counter);
                counter++;
            }
        }

        for (int t = 0; t < lofRuns.length; t++) {
            lofRuns[t].sort(lofRuns[t].numAttributes() - 2);    // sorts by ascending order of LOF
        }

        Instances newData = new Instances(data);
        Attribute newLOFAttr = new Attribute("BreadthFirstLOF");
        newData.insertAttributeAt(newLOFAttr, newData.numAttributes());

        boolean[] inFinalRanking = new boolean[data.numInstances()];
        int finalRank = data.numInstances();      // higher needs to be more outlier-ish, so start high and count down to 1


        for (int posInMiniRanking = data.numInstances() - 1; posInMiniRanking >= 0; posInMiniRanking--) {
            for (int t = 0; t < lofRuns.length; t++) {

                // Have we used item posInMiniRanking in lofRuns[t] yet?
                int itemIndex = (int) lofRuns[t].instance(posInMiniRanking).value(lofRuns[t].numAttributes() - 1);

                if (! inFinalRanking[itemIndex]) {
                    // get to itemIndex in newData
                    Instance inst = newData.instance(itemIndex);
                    // assign it finalRank
                    inst.setValue(newData.numAttributes() - 1, finalRank);

                    // update finalRank and inFinalRanking
                    finalRank--;
                    inFinalRanking[itemIndex] = true;

                }
            }
        }

        return newData;
    }

    // Code found on http://introcs.cs.princeton.edu/java/14array/Sample.java.html
    // Returns an array of 0-based indices. The sample is found in the first numElements of that array.
    public static int[] sampleWithoutReplacement(int numWanted, int numElements, Random randomness) {

        // create permutation 0, ..., numElements-1
        int[] perm = new int[numElements];
        for (int i = 0; i < numElements; i++)
            perm[i] = i;

        // create random sample in perm[0], perm[1], ..., perm[numWanted-1]
        for (int i = 0; i < numWanted; i++)  {

            // random integer between i and numElements-1
            int r = i + randomness.nextInt(numElements-i);

            // swap elements at indices i and r
            int t = perm[r];
            perm[r] = perm[i];
            perm[i] = t;
        }

        return(perm);
        
    }


}