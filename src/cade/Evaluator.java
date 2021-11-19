package cade;

import com.github.rcaller.rstuff.RCaller;
import com.github.rcaller.rstuff.RCode;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.core.Instances;

import java.util.ArrayList;


public class Evaluator {

    public static double getAUCFromWeka(ArrayList<NominalPrediction> predArray) {
        Instances result = new ThresholdCurve().getCurve((ArrayList<Prediction>) predArray.clone());
        return ThresholdCurve.getROCArea(result);
    }

    //convert the predictions into rankings and return Spearman's rank correlation
    //Calls R. Assumes both one and two are the same length.
    public static double computeAndCompareRankings2(ArrayList<NominalPrediction> one, ArrayList<Double> two) {
        //Convert one and two into string arrays of predictions
        double[] onePreds = new double[one.size()];
        String oneStr = "c(", twoStr = "c(";
        for (int i = 0; i < onePreds.length; i++) {
            onePreds[i] = one.get(i).distribution()[0];
            if (i != 0) {
                oneStr += ",";
                twoStr += ",";
            }
            oneStr += onePreds[i];
            twoStr += two.get(i);
        }
        oneStr += ")";
        twoStr += ")";

        //now use R to convert them into rankings and compare them
        RCaller caller = RCaller.create();
        RCode code = RCode.create();
        code.addRCode("one = " + oneStr);
        code.addRCode("two = " + twoStr);
        code.addRCode("oneRank = rank(one)");
        code.addRCode("twoRank = rank(two)");
        code.addRCode("result = cor(oneRank, twoRank, method=\"spearman\")");
        caller.setRCode(code);
        caller.runAndReturnResult("result");
        double[] result = caller.getParser().getAsDoubleArray("result");
        return result[0];
    }


}
