package cade;

import weka.classifiers.evaluation.NominalPrediction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;

// Useful for diagnostics
public class StatsZeroCounter {

    public int numPosInstSeen, numNegInstSeen, totNumInstSeen;

    int numClassifierZeroes;
    int numClassifierOnes;
    int numDensityZeroes;
    int numFinalZeroes;
    int numTiesBefore;
    double tieValueBefore;
    int numTiesAfter;


    // Note: when we use smoothing, zeroes go away. Turn it off to count them most easily.
    public void printSummaryOfPredictions() {
        System.out.println("Instances seen: " + totNumInstSeen + " total; " +
                numPosInstSeen + " pos, " + numNegInstSeen + " neg");
        System.out.println("Density estimate 0's: " + numDensityZeroes);
        System.out.println("(Number of tied instances shown is the sum across all folds, and it's often misleading for naive Bayes)");
    }

    public void printOneClassifierSummary(int classifierID) {
        System.out.println("Classifier " + classifierID + ": 0's: " + numClassifierZeroes + ", 1's: " + numClassifierOnes
                + ", final 0's: " + numFinalZeroes);
        System.out.println("Classifier " + classifierID + " tied values: " + numTiesBefore + ", having P(+)=" + tieValueBefore
                + " (for example), final num ties: " + numTiesAfter);
    }

    public void countZeroes(Instances testInstances, double[][] testSetPredictions, double[] densityLogEstimate) {

        for (int i = 0; i < testInstances.numInstances(); i++) {
            Instance inst = testInstances.instance(i);
            totNumInstSeen++;
            if (inst.classAttribute().value((int) inst.classValue()).equals(Parameters.positiveClassLabel))
                numPosInstSeen++;
            else
                numNegInstSeen++;


            if (testSetPredictions[i][0] == 0)
                numClassifierZeroes++;
            else if (testSetPredictions[i][1] == 0)    // better to test equality to 0 than to 1
                numClassifierOnes++;

            if (Double.isInfinite(densityLogEstimate[i]) && densityLogEstimate[i] < 0)  // i.e., density estimate == 0
                numDensityZeroes++;

            if ((Double.isInfinite(densityLogEstimate[i]) && densityLogEstimate[i] < 0) || testSetPredictions[i][0] == 0)
                numFinalZeroes++;

        }
    }

    public void countTieScores(double[][] classifierPredictions, ArrayList<NominalPrediction> finalPredictions) {

        // I'd like to be sure to handle 1's that aren't actually 1's correctly, but alas, I can't see a way to do so without
        // writing my own sort function.

        // Check the "before" scores from the classifier
        double[] scoresBefore = new double[classifierPredictions.length];
        for (int i = 0; i < classifierPredictions.length; i++) {
            scoresBefore[i] = classifierPredictions[i][0];
        }

        // count ties
        double[] tiesBefore = findLongestTieValues(scoresBefore);

        // Check the "after" scores
        double[] scoresAfter = new double[finalPredictions.size()];
        for (int i = 0; i < finalPredictions.size(); i++) {
            scoresAfter[i] = finalPredictions.get(i).distribution()[0];
        }
        double[] tiesAfter = findLongestTieValues(scoresAfter);

        numTiesBefore += (int) tiesBefore[0];
        tieValueBefore = tiesBefore[1];         // note: overwrites value from the previous fold. oh well--it's just an example.
        numTiesAfter += (int) tiesAfter[0];

    }

    // returns two values: {maxRunLength, maxRunValue}
    public static double[] findLongestTieValues(double[] scores) {

        Arrays.sort(scores);

        // initialize so we're coming off a run of length 0 at the first value. This way, first thing that happens
        // is to record a run-in-progress of length 1 at the first value in the array.
        int prevRunLength = 0;
        double prevValue = scores[0];
        int maxRunLength = 0;
        double maxRunValue = scores[0] - 1;
        for (int i = 0; i < scores.length; i++) {
            if (scores[i] == prevValue) {
                prevRunLength++;
            } else {
                // this run of ties is over. How was it?
                if (prevRunLength > maxRunLength) {
                    maxRunLength = prevRunLength;
                    maxRunValue = prevValue;
                }

                prevValue = scores[i];
                prevRunLength = 1;
            }

        }
        // at the end, end that run properly
        if (prevRunLength > maxRunLength) {
            maxRunLength = prevRunLength;
            maxRunValue = prevValue;
        }

        return new double[]{maxRunLength, maxRunValue};
    }


}

