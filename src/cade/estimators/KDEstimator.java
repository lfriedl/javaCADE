package cade.estimators;

import java.io.File;
import java.io.IOException;

import cade.Parameters;
import com.github.rcaller.rstuff.RCaller;
import com.github.rcaller.rstuff.RCode;
import weka.core.Instances;


public class KDEstimator extends Estimator {
	boolean loadTrainingDataOnce = true;
	public String fileName = "";
	public static double totalTime0 = 0;
	public static double totalTime1 = 0;
	public static double totalTime2 = 0;
	public static int numTimes = 0;

    double[] values;


    public KDEstimator(Instances instances, int attrNum) {
        values = valuesWithoutMissing(instances, attrNum);
        if (loadTrainingDataOnce){
            createAndSaveEstimator();
        }
    }

    // Used in testing
    public KDEstimator(double[] values) {
        this.values = values;
        if (loadTrainingDataOnce) {
            createAndSaveEstimator();
        }
    }

    protected void createAndSaveEstimator() {
        try {
//            File systemTmpDir = new File("/tmp");
//            File file = File.createTempFile("CADE", ".Rdata", systemTmpDir);
            File file = File.createTempFile("CADE", ".Rdata");
            fileName = file.getPath();

            RCaller rcaller = Parameters.rCaller;
            RCode code = Parameters.rCode;
            code.clearOnline();

            code.addRCode("library(ks);");
            code.addRCode("library(KernSmooth);");

            code.addRCode("ptm <- proc.time()");

            code.addDoubleArray("train_x" , values);
            code.addRCode("kde = kde(x=train_x, binned=F, supp=20, " +
                    "h=tryCatch(hpi(train_x), error=function(e) { dpik(train_x, scalest=\"stdev\") }) )");
            code.addRCode("save(kde, file=\"" + fileName + "\")");

            code.addRCode("totalTime <- c(proc.time() - ptm)");

            rcaller.runAndReturnResultOnline("totalTime");

            double[] result = rcaller.getParser().getAsDoubleArray("totalTime");
            totalTime2 += result[2];
            totalTime1 += result[1];
            totalTime0 += result[0];
            numTimes++;

        } catch (IOException e) {
            System.out.println("Failed to create temp file");
            e.printStackTrace();
        }
    }

    /**
     * Example from library(ks) documentation:
     * x <- rnorm.mixt(n=10000, mus=0, sigmas=1, props=1)
     * fhat <- kde(x=x, h=hpi(x))
     * To get the density, it should be:
     * dens = dkde(x, fhat)
     *
     * However, sometimes, hpi(x) fails with this warning:
     *   "scale estimate is zero for input data"
     * In that case, a reasonable thing to do seems to be to run dpik (which is being called anyway)
     * with an alternative flag: (using scalest="stdev" will return something as long as there's more than 1 value)
     * dpik(x, scalest="stdev")
     * Wrapped up with error handling, that gives:
     * h = tryCatch(hpi(x), error=function(e) { dpik(x, scalest="stdev") })
     *
     * [Modified Oct 2021 because of changes to kde() function. Added flags "supp=20, binned=F" when constructing it.]
     */
    public double[] probabilityOf(double[] attrVals) {
        RCaller rcaller = Parameters.rCaller;
        RCode code = Parameters.rCode;
        code.clearOnline();
        code.addRCode("library(ks)");
        code.addRCode("library(KernSmooth)");

        code.addDoubleArray("test_y", attrVals);
      
        if (loadTrainingDataOnce){
        	code.addRCode("load(\"" + fileName + "\")");
        	code.addRCode("densVal = dkde(test_y, kde)");
        } else {
        	code.addDoubleArray("train_x" , values);
            code.addRCode("densVal = dkde(test_y, kde(x=train_x, binned=F, supp=20, " +
                    "h=tryCatch(hpi(train_x), error=function(e) { dpik(train_x, scalest=\"stdev\") }) ))");
        }

        rcaller.runAndReturnResultOnline("densVal");
        double[] result = rcaller.getParser().getAsDoubleArray("densVal");
        return result;
    }

}
