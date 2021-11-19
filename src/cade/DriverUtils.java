package cade;

import java.io.BufferedWriter;
import java.io.IOException;

public class DriverUtils {

    // Takes a comma-separated list of numbers, and returns the same thing in an int array
    public static int[] convertStringToIntArray(String attrsOneString) {

        if (attrsOneString.length() < 1) {
            return new int[]{};
        }

        String[] attrStrings = attrsOneString.split(",");
        int[] retVal = new int[attrStrings.length];
        for (int i = 0; i < attrStrings.length; i++) {
            retVal[i] = Integer.parseInt(attrStrings[i]);
        }
        return retVal;
    }

    public static String convertBackToString(int[] attrs) {
        StringBuffer sb = new StringBuffer("{ ");
        for (int attr : attrs) {
            sb.append(attr + ",");
        }
        sb.append(" }");
        return sb.toString();
    }

    // avg & var for LOF trials in data structure: lofResults[runNum][lofMethod]
    public static void printLOFResults(double[][] lofResults, BufferedWriter writer) throws IOException {
        for (int lofMethod = 0; lofMethod < lofResults[0].length; lofMethod++) {
            double sum = 0;
            double sumOfSquares = 0;
            double divideBy = lofResults.length;
            System.out.print("avg-var LOF method" + lofMethod + ": ");
            if (writer != null)
                writer.write("avg-var LOF method" + lofMethod + ": ");

            for (int runNum = 0; runNum < lofResults.length; runNum++) {
                if (lofResults[runNum][lofMethod] <= 1 && lofResults[runNum][lofMethod] >= 0) {
                    sum += lofResults[runNum][lofMethod];
                    sumOfSquares += Math.pow(lofResults[runNum][lofMethod], 2);
                } else
                    divideBy--;
            }
            double mean = sum / divideBy;

            System.out.print(mean + " ");//print mean
            System.out.println((sumOfSquares / divideBy) - (mean * mean));//print variance

            if (writer != null)
                writer.write(mean + " " + ((sumOfSquares / divideBy) - (mean * mean)) + "\n");

        }
    }

    // avg and var for CADE trials in data structure: aucs[pNegMethod][classifier][runNum]
    public static void printCADEResults(double[][][] aucs, BufferedWriter writer, ParamsMultiRuns paramsMeta) throws IOException {
        //average over all trials per setting and print
        for (int pNegMethod = 0; pNegMethod < aucs.length; pNegMethod++) {
            Parameters.ClassifierType[] allClassifiersRun = paramsMeta.getAllClassifierTypesThisRun();
            String[] classifierNames = new String[allClassifiersRun.length];
            for (int i = 0; i < classifierNames.length; i++) {
                classifierNames[i] = "" + allClassifiersRun[i];
//                if (paramsMeta.varyClassifierByIteration) {
//                    for (int j = 1; j < paramsMeta.numIterations; j++)
//                        classifierNames[i] += ", " + paramsMeta.classifierTypesByIteration[j][i];
//                }
            }

            for (int classifier = 0; classifier < aucs[pNegMethod].length; classifier++) {
                double sum = 0;
                double sumOfSquares = 0;
                double divideBy = aucs[pNegMethod][classifier].length;
                System.out.print("avg-var " + paramsMeta.pseudoNegGenerationTypesThisRun[pNegMethod] + " " + classifierNames[classifier] + ": ");
                if (writer != null)
                    writer.write("avg-var " + paramsMeta.pseudoNegGenerationTypesThisRun[pNegMethod] + " " + classifierNames[classifier] + ": ");
                for (int runNum = 0; runNum < paramsMeta.numRunsPerSetting; runNum++) {
                    if (aucs[pNegMethod][classifier][runNum] <= 1 && aucs[pNegMethod][classifier][runNum] >= 0) {
                        sum += aucs[pNegMethod][classifier][runNum];
                        sumOfSquares += Math.pow(aucs[pNegMethod][classifier][runNum], 2);
                    } else
                        divideBy--;

                }
                double mean = sum / divideBy;

                System.out.print(mean + " ");//print mean
                System.out.println((sumOfSquares / divideBy) - (mean * mean));//print variance

                if (writer != null)
                    writer.write(mean + " " + ((sumOfSquares / divideBy) - (mean * mean)) + "\n");
            }
        }
    }

}
