package cade.positiveDataGenerators;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Random;

import cade.Parameters;
import com.github.rcaller.rstuff.RCaller;
import com.github.rcaller.rstuff.RCode;
import weka.core.*;


public class GaussianMixtureDataGenerator extends DataGenerator {
    double[][] posMeans;
    double[][][] posCovars;
    int numGaussians = 1;
    int[] meanRange = {-10, 10};
    Random rng;
    boolean debugMode = false;
    double[] avgAbsCorr;
    String covMethod = "eigen";
    int numGaussianDims;
    int numNonsenseDims = 0; //num nonsense dimensions on top of the # passed in
    int[] nonsenseRange = {-20, 20};//each nonsense dim will be uniform within this range
    boolean randomCovarMethods = false;
    boolean runWithSameGaussians = false;

    public GaussianMixtureDataGenerator(int numDimensions, int numNonsenseDims, int numTrainingPositives) {
        this.numTrainingPositives = numTrainingPositives;
        numAttributes = numDimensions;
        numGaussianDims = numAttributes;
        this.numNonsenseDims = numNonsenseDims;
        numAttributes += numNonsenseDims;

        rng = new Random();
        loadData();
        calculatePropertiesFromData();
    }

    public GaussianMixtureDataGenerator(int numDimensions, int numTrainingPositives, String covMethod) {
        this.numTrainingPositives = numTrainingPositives;
        this.covMethod = covMethod;
        numAttributes = numDimensions;
        numGaussianDims = numAttributes;
        rng = new Random();
        loadData();
        calculatePropertiesFromData();
    }


    //Generate random gaussian mixture models
    @Override
    public void loadData() {

        posMeans = new double[numGaussians][numGaussianDims];
        posCovars = new double[numGaussians][numGaussianDims][numGaussianDims];
        avgAbsCorr = new double[numGaussians];

        //randomly generate positive means
        for (int i = 0; i < numGaussians; i++)
            for (int j = 0; j < numGaussianDims; j++)
                posMeans[i][j] = rng.nextInt(meanRange[1] - meanRange[0] + 1) + meanRange[0];

        if (runWithSameGaussians) {
            Random temp = new Random(1);
            for (int i = 0; i < numGaussians; i++)
                for (int j = 0; j < numGaussianDims; j++)
                    posMeans[i][j] = temp.nextInt(meanRange[1] - meanRange[0] + 1) + meanRange[0];
        }


        //randomly generate covariance matrices
        for (int g = 0; g < numGaussians; g++) {
            RCaller caller = RCaller.create();
            RCode code = RCode.create();

            code.addRCode("library(clusterGeneration)");
            String[] covMethodOptions = new String[]{"eigen", "onion", "c-vine", "unifcorrmat"};
            if (randomCovarMethods) {
                Random tempRng = new Random();
                covMethod = covMethodOptions[tempRng.nextInt(4)];
            }

            if (runWithSameGaussians) {
                code.addRCode("load(\"~agentzel/mat" + g + ".Rdata\")");
            } else
                code.addRCode("mat" + g + " = genPositiveDefMat(" + numGaussianDims + ", covMethod=\"" + covMethod + "\")");
            caller.setRCode(code);
            caller.runAndReturnResult("mat" + g);

            System.out.println("covMethod = " + covMethod);

            double[][] results = caller.getParser().getAsDoubleMatrix("Sigma", numGaussianDims, numGaussianDims);
            posCovars[g] = results;

            //calculate average of the absolute values of the correlation for each matrix
            int count = 0;
            for (int x = 0; x < numGaussianDims; x++) {
                for (int y = x + 1; y < numGaussianDims; y++) {
                    avgAbsCorr[g] += Math.abs(posCovars[g][x][y] / (Math.sqrt(posCovars[g][x][x]) * Math.sqrt(posCovars[g][y][y])));
                    count++;
                }
            }
            avgAbsCorr[g] /= count;

            System.out.println(avgAbsCorr[g]);
            System.out.println(posMeans[g][0] + " " + posMeans[g][1]);
        }

    }

    @Override
    public void calculatePropertiesFromData() {
        posStdDevs = null;
    }


    //creates numInstances instances with label from the given means and covars
    public Instances generateLabeledPositives(int numInstances, String label) {
        Instances insts = generateUnlabeledInstances(numInstances);

        // Add a new attribute to hold the class label
        FastVector classVals = new FastVector(2);
        classVals.addElement(Parameters.positiveClassLabel);
        classVals.addElement(Parameters.negativeClassLabel);

        Attribute classAttr = new Attribute("classAttr", classVals);

        insts.insertAttributeAt(classAttr, numAttributes);


        insts.setClassIndex(insts.numAttributes() - 1);

        for (int i = 0; i < insts.numInstances(); i++) {
            Instance inst = insts.instance(i);
            inst.setClassValue(label);
        }

        return insts;

    }

    public Instances generateUnlabeledInstances(int numInstances) {
//    	//kinda a hack - for generating pseudo negatives, it automatically calls with numInstances = -1, so this...gets around that
//    	if (numInstances <= 0)
//    		numInstances = numTrainingPositives;
//    		
        // Convert double array to Instances
        FastVector attributes = new FastVector(numGaussianDims);
        for (int i = 0; i < numGaussianDims; i++) {
            Attribute tempAttr = new Attribute("attr" + i); // I assume we'll never need the same later
            attributes.addElement(tempAttr);
        }


        Instances[] separatedInstances = new Instances[numGaussians];
        for (int g = 0; g < numGaussians; g++) {
            //returns what we want, but as a matrix instead of instances
            double[][] dataMatrix = generateDataMatrixInR(posMeans[g], posCovars[g], numInstances);

            Instances instances = new Instances("R-generated data", attributes, numInstances);

            for (int i = 0; i < dataMatrix.length; i++) {
                Instance tempInst = new DenseInstance(1, dataMatrix[i]);
                tempInst.setDataset(instances);
                instances.add(tempInst);
            }

            separatedInstances[g] = instances;
        }


        //Now, sample numInstances instances from the groups of instances we have
        Instances instances = new Instances("R-generated data", attributes, numInstances);

        int[] currInst = new int[numGaussians];
        for (int i = 0; i < numInstances; i++) {
            int whichGaussian = rng.nextInt(numGaussians);
            instances.add(separatedInstances[whichGaussian].instance(currInst[whichGaussian]));
            currInst[whichGaussian]++;
        }

        //Now, if we need nonsenseDims, add those in
        for (int i = 0; i < numNonsenseDims; i++) {
            instances.insertAttributeAt(new Attribute("Nonsense" + (i + 1)), numGaussianDims + i);
            for (int j = 0; j < instances.numInstances(); j++) {
                instances.instance(j).setValue(numGaussianDims + i, rng.nextInt(1 + nonsenseRange[1] - nonsenseRange[0]) + nonsenseRange[0]);
            }
        }

        return instances;
    }

    // Later, this might create other types of things besides normals.
    public double[][] generateDataMatrixInR(double[] means, double[][] covars, int numInstances) {

        String rMeans = "c(";
        for (int i = 0; i < means.length; i++) {
            if (i != 0)
                rMeans += ", ";
            rMeans += means[i];
        }
        rMeans += ")";
        String rCovar = "c(";

        for (int col = 0; col < numGaussianDims; col++)
            for (int row = 0; row < numGaussianDims; row++) {
                if (col != 0 || row != 0)
                    rCovar += ", ";
                rCovar += covars[row][col];
            }

        rCovar += ")";

        return generateNormals(numInstances, rMeans, rCovar);
    }

    public double[][] generateNormals(int numInstances, String rMeans, String rCovarMatrix) {
        RCaller caller = RCaller.create();
        RCode code = RCode.create();

        code.addRCode("library(mnormt)");
        // Example string: "varcov = matrix(c(.5, .9, .9, 5), nrow=2)"
        code.addRCode("varcov = matrix(" + rCovarMatrix + ", nrow=" + numGaussianDims + ")");
        // Example string: "x = rmnorm(n=10, mean = c(0, 100), varcov=varcov)"
        code.addRCode("x = rmnorm(n=" + numInstances + ", mean = " +
                rMeans + ", varcov=varcov)");
        caller.setRCode(code);
        caller.runAndReturnResult("x");

        double[][] results;

        results = caller.getParser().getAsDoubleMatrix("x", numInstances, numGaussianDims);

        if (debugMode) {
            for (int i = 0; i < results.length; i++) {
                String tmp = "results[" + i + ",] : ";
                for (int j = 0; j < results[i].length; j++) {
                    tmp = tmp + results[i][j] + " ";
                }
                System.out.println(tmp);
            }
        }

        return results;
    }


    public Instances generateTrainingPositives(int numInstances) {
        Instances insts = generateLabeledPositives(numInstances, Parameters.positiveClassLabel);

        if (insts.numInstances() <= 1)
            return insts;

        min = new double[numAttributes];
        max = new double[numAttributes];
        for (int i = 0; i < numAttributes; i++) {
            if (!insts.attribute(i).isNumeric()) {
                min[i] = -1;
                max[i] = -1;
            } else {
                double currMin, currMax;
                currMin = insts.instance(0).value(i);
                currMax = insts.instance(0).value(i);
                for (int j = 1; j < insts.numInstances(); j++) {
                    if (insts.instance(j).value(i) < currMin)
                        currMin = insts.instance(j).value(i);
                    if (insts.instance(j).value(i) > currMax)
                        currMax = insts.instance(j).value(i);
                }
                min[i] = currMin;
                max[i] = currMax;

            }
        }
        trainingSample = insts;

        return insts;
    }

    @Override
    public Instances generateTrainingNegatives(int numInstances) {
        System.err.println("Invalid attempt to generate training negatives - implement me!");
        return null;
    }

    @Override
    public Instances generateTestingPositives(int numInstances) {
        return generateLabeledPositives(numInstances, Parameters.positiveClassLabel);
    }

    @Override
    public Instances generateTestingNegatives(int numInstances) {

        Instances data = generateTrainingPositives(1);
        Instances instances = new Instances(data, numInstances);//creates an empty data set using header info from data

        for (int currInst = 0; currInst < numInstances; currInst++) { //create numInstances instances
            Instance dataPoint = new DenseInstance(data.instance(0));
            dataPoint.setDataset(instances);
            for (int i = 0; i < numAttributes; i++) {    //loop over all the attributes
                //get a random # between the min and max for this attribute
                double val = rng.nextDouble() * (max[i] - min[i]) + min[i];
                dataPoint.setValue(i, val);
            }
            instances.add(dataPoint);
        }
        return instances;
    }

    public String printGaussians() {
        String str = "";
        for (int i = 0; i < numGaussians; i++) {
            if (i != 0)
                str += "\n";
            str += printArray(posMeans[i]) + "\n";
            str += printCovar(posCovars[i]) + "\n";
        }
        return str;
    }

    private String printArray(double[] array) {
        String val = "(";
        for (int i = 0; i < array.length; i++) {
            if (i != 0)
                val += ", ";
            val += array[i];
        }
        val += ")";

        return val;
    }

    private String printCovar(double[][] covar) {
        String val = "";

        for (int i = 0; i < covar.length; i++) {
            for (int j = 0; j < covar.length; j++) {
                if (j != 0)
                    val += " ";
                val += covar[i][j];
            }
            val += "\n";
        }
        return val;
    }

    public String printAvgCorr() {
        String result = "(";

        for (int i = 0; i < numGaussians; i++) {
            if (i != 0)
                result += ", ";
            result += avgAbsCorr[i];
        }

        result += ")";

        return result;
    }

    //gets the actual probability density at each point in instances
    public ArrayList<Double> getTrueProbabilities(Instances instances) {
        ArrayList<Double> predictions = new ArrayList<Double>();


        //first, convert the instances into a double matrix string for R
        String instString = "matrix(c(";
        for (int attr = 0; attr < numGaussianDims; attr++)
            for (int inst = 0; inst < instances.numInstances(); inst++) {
                if (attr != 0 || inst != 0)
                    instString += ",";
                instString += instances.instance(inst).value(attr);
            }
        instString += "), nrow=" + instances.numInstances() + ", ncol=" + numGaussianDims + ")";

        double[][] individualPredictions = new double[numGaussians][instances.numInstances()];
        for (int g = 0; g < numGaussians; g++) {
            //For each gaussian, get the predictions on all the instances
            String meanVector = "c(";

            for (int i = 0; i < posMeans[g].length; i++) {
                if (i != 0)
                    meanVector += ",";
                meanVector += posMeans[g][i];
            }

            meanVector += ")";

            String varcov = "matrix(c(";
            for (int i = 0; i < posCovars[g].length; i++) {
                for (int j = 0; j < posCovars[g][i].length; j++) {
                    if (i != 0 || j != 0)
                        varcov += ",";
                    //hack to fix rounding issue that was making matrix nonsymmetric
                    if (i < j)
                        varcov += posCovars[g][i][j];
                    else
                        varcov += posCovars[g][j][i];
                }
            }

            varcov += "), nrow=" + numGaussianDims + ", ncol=" + numGaussianDims + ")";

            //Now, R calling time!
            RCaller caller = RCaller.create();
            RCode code = RCode.create();


            code.addRCode("library(mnormt)");
            code.addRCode("result = dmnorm(x=" + instString + ", mean=" + meanVector + ", varcov=" + varcov + ")");
            caller.setRCode(code);
            caller.runAndReturnResult("result");


            individualPredictions[g] = caller.getParser().getAsDoubleArray("result");
//            files[0].delete();
//            files[1].delete();
        }

        //now combine predictions
        for (int i = 0; i < instances.numInstances(); i++) {
            predictions.add(i, 0.0);
            for (int g = 0; g < numGaussians; g++)
                predictions.set(i, predictions.get(i) + ((double) 1 / numGaussians) * individualPredictions[g][i]);
        }

        return predictions;
    }

    public String printAvgAbsCorr() {
        DecimalFormat df = new DecimalFormat("0.000");

        String result = "(";
        for (int i = 0; i < avgAbsCorr.length; i++) {
            if (i != 0)
                result += ", ";
            result += df.format(avgAbsCorr[i]);
        }

        result += ")";
        return result;
    }

    //prints a single number for all gaussians - the average of the avgAbsCorr for each one
    public String printAvgAvgAbsCorr() {
        DecimalFormat df = new DecimalFormat("0.000");

        double avg = 0;
        for (int i = 0; i < numGaussians; i++)
            avg += avgAbsCorr[i];
        avg /= numGaussians;

        return df.format(avg);
    }
}
