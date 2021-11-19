package cade.generativeProbDensities;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import cade.Parameters;
import com.github.rcaller.rstuff.RCaller;
import com.github.rcaller.rstuff.RCode;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import cade.estimators.Estimator;

// This class can't quite fit in the usual hierarchy, using Estimators (alas), because the Estimator class is only
// built for a single attribute at a time.
public class BayesNetProbDensity extends ProbDensity{
	File numericTrainingFile;
	File factorTrainingFile;
	public String learnedBayesNetFileNameN = "";

	boolean[] isNumeric;	// Every attribute is either numeric or "factor" (a.k.a. categorical, nominal)
	boolean[] constAttrs;
	double[] constValues;//values of constant attributes (constValues[i] only meaningful if constAttrs[i] is true)
	ArrayList<Integer> numericAttrs;
	ArrayList<Integer> factorAttrs;
	boolean hasNumeric, hasFactor;
	int numNumeric, numFactor, numConst;
	Instance exampleInstance;
	
	int numAtWhichToSubdivide = 10000;
	int subdivisionSize = 10000;
	boolean structurelessBayesNet = false;

	public BayesNetProbDensity(Parameters params) {
		super(params);
		try {
			characterizeAttrTypes();
			writeTrainingDataToFiles();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	// Sets up: constAttrs, constValues, factorAttrs, isNumeric (<-- may change this one)
	private void characterizeAttrTypes() {
		exampleInstance = dataGen.trainingSample.firstInstance();

		//figure out what type of attributes we have.
		// 3 types are relevant for the Bayes net: numeric; factor; constant across all insts.

		//figure out which attributes are constant in the training data and must be skipped over
		numConst = 0;
		constAttrs = new boolean[exampleInstance.numAttributes()-1];
		constValues = new double[exampleInstance.numAttributes()-1];
		for (int i = 0; i < exampleInstance.numAttributes()-1; i++){
			int numDistinctValues = dataGen.trainingSample.numDistinctValues(i);
			if (numDistinctValues == 1) {
				constAttrs[i] = true;
				constValues[i] = exampleInstance.value(i);
				numConst++;
			}
		}

		// Construct lists of numeric & factor attrs
		isNumeric = new boolean[exampleInstance.numAttributes()-1];
		numericAttrs = new ArrayList<Integer>();
		factorAttrs = new ArrayList<Integer>();
		for (int i = 0; i < exampleInstance.numAttributes()-1; i++) {
			if (!constAttrs[i]) {
				if (exampleInstance.attribute(i).type() == 0) {
					isNumeric[i] = true;
					numericAttrs.add(i);
				} else {
					factorAttrs.add(i);
				}
			}
		}
		numNumeric = numericAttrs.size();
		hasNumeric = (numNumeric > 0);

		numFactor = factorAttrs.size();	//num factor attributes
		hasFactor = (numFactor > 0);
	}

	private void writeTrainingDataToFiles() throws IOException{
		Instances posTrainInstances = dataGen.trainingSample;

		numericTrainingFile = File.createTempFile("numericTraining", ".csv");
		factorTrainingFile = File.createTempFile("factorTraining", ".csv");

		if (hasNumeric) {
			writeAttrsToFile(posTrainInstances, numericTrainingFile, numericAttrs);
		}

		if (hasFactor) {
			writeAttrsToFile(posTrainInstances, factorTrainingFile, factorAttrs);
		}

	}

	protected void writeAttrsToFile(Instances instances, File fileObj, ArrayList<Integer> attrIndices) throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter(fileObj));
		for (int i = 0; i < instances.numInstances(); i++){
			boolean somethingWritten = false;//help with placing commas
			Instance inst = instances.instance(i);
			for (int attrIndex : attrIndices) {
				if (somethingWritten)
					writer.write(",");
				writer.write("" + inst.value(attrIndex));
				somethingWritten = true;
				hasFactor = true;

			}
			writer.write("\n");
		}
		writer.flush();
	}


	public Instances generateItems(int numInstances) throws Exception {
        Instances newInstances = new Instances(exampleInstance.dataset(), numInstances);//creates an empty data set using header info from data

		if (numInstances == -1)
			numInstances = dataGen.numTrainingPositives();
        else if (numInstances == 0){
        	System.out.println("UH OH!!!");
        	return newInstances;
        }

		double[][] numericInstances = null;
		double[][] factorInstances = null;

		RCaller caller = Parameters.rCaller;
		RCode code = Parameters.rCode;

		if (hasNumeric){
			String varNames = "c(";
	       	 for (int i = 0; i < numNumeric; i++){
	       		 if (i != 0) varNames += ", ";
	       		 varNames += "\"V" + (i+1)+"\"";
	       	 }
	       	 varNames += ")";

			code.clearOnline();
			code.addRCode("library(bnlearn)");
			
			code.addRCode("numericTraining = read.csv('" + numericTrainingFile.getAbsolutePath() + "', header=FALSE, colClasses=\"numeric\")");
			if (structurelessBayesNet)
				code.addRCode("res = empty.graph("+varNames+")");
			else
				code.addRCode("res = mmhc(numericTraining)");
			
			code.addRCode("fit = bn.fit(res, numericTraining)");
	
			//now save learned bayes net so we don't have to re-learn it later
			File file = File.createTempFile("CADE-BN", ".Rdata");

			learnedBayesNetFileNameN = file.getPath();
			code.addRCode("save(fit, file=\"" + learnedBayesNetFileNameN + "\")");
			
			code.addRCode("insts = as.matrix(rbn(fit, n=" + numInstances + ", debug=TRUE))");
			code.addRCode("insts = t(insts)");
			caller.runAndReturnResultOnline("insts");
			numericInstances = caller.getParser().getAsDoubleMatrix("insts", numInstances, numNumeric);

		} if (hasFactor){
			String varNames = "c(";
	       	 for (int i = 0; i < numFactor; i++){
	       		 if (i != 0) varNames += ", ";
	       		 varNames += "\"V" + (i+1)+"\"";
	       	 }
	       	 varNames += ")";

			code.clearOnline();
			code.addRCode("library(bnlearn)");
       	 
            code.addRCode("factorTraining = read.csv('" + factorTrainingFile.getAbsolutePath() + "', header=FALSE, colClasses=\"factor\")");
           
            if (structurelessBayesNet)
				code.addRCode("res = empty.graph("+varNames+")");
			else
				code.addRCode("res = mmhc(factorTraining)");
           
            if (!useSmoothing)
            	code.addRCode("fit = bn.fit(res, factorTraining)"); 
            else{
            	code.addRCode("fitTemp = bn.fit(res, factorTraining)");
            	//now we need to smooth to avoid 0s
            	code.addRCode("source(\"./R-code/smooth.R\")");
            	code.addRCode("fit = bn.smooth(net=fitTemp)");
            }            

            code.addRCode("insts = as.matrix(rbn(fit, n=" + numInstances + "))");
            code.addRCode("insts = matrix(as.numeric(insts), nrow=nrow(insts))");
            code.addRCode("insts = t(insts)");

            caller.runAndReturnResultOnline("insts");
            factorInstances = caller.getParser().getAsDoubleMatrix("insts", numInstances, numFactor);

		}
        
        //potentially, combine both factor and numeric attributes to create whole Instances
        for (int currInst = 0; currInst < numInstances; currInst++) { //create numInstances instances
            Instance newDataPoint = new DenseInstance(exampleInstance.numAttributes());
            newDataPoint.setDataset(newInstances);
            //used to keep track of which numeric/factor attribute value we're on
            int numericIndex = 0;
            int factorIndex = 0;
            for (int attr = 0; attr < numNumeric+numFactor+numConst; attr++){
            	if (constAttrs[attr])
            		newDataPoint.setValue(attr, constValues[attr]);
            	else if (isNumeric[attr]){
            		newDataPoint.setValue(attr, numericInstances[currInst][numericIndex]);
            		numericIndex++;
            	} else {
            		newDataPoint.setValue(attr, factorInstances[currInst][factorIndex]);
            		factorIndex++;
            	}
            }
            newInstances.add(newDataPoint);
        }
        
        //now, label as negative
        newInstances.setClassIndex(newInstances.numAttributes() - 1);
        setClassLabelNegative(newInstances);
        
		return newInstances;
	}

	public Estimator[] constructEstimators() {
		return null;
	}
	
	public double[] computeLogDensity(Instances testInstances){
		
		// These arrays range along the testInstances
        double[] currentLogProbs = new double[testInstances.numInstances()];
        
        System.out.println(testInstances.numInstances() + " test instances");
        if (testInstances.numInstances() < numAtWhichToSubdivide){
	        try {
	        	currentLogProbs = getPredsFromBayesNet(testInstances);
	        } catch (IOException e) {
	        	System.err.println("AHHHHHH!!!!!!!!!");
	        	e.printStackTrace();
	        }
        } else {//we need to subdivide testInstances and send it in batches
        	int numInstancesRemaining = testInstances.numInstances();
        	int numInstancesCompleted = 0;
        	while (numInstancesRemaining > 0){
        		System.out.println(numInstancesRemaining + " left");
        		
        		int currSubdivisionSize = Math.min(subdivisionSize, numInstancesRemaining);
	        	Instances currTestInstances = new Instances(testInstances, currSubdivisionSize);
	        	numInstancesRemaining -= currSubdivisionSize;
	        	for (int i = 0; i < currSubdivisionSize; i++)
	        		currTestInstances.add(testInstances.instance(numInstancesCompleted+i));
	        	
	        	double[] smallerLogProbs = new double[currSubdivisionSize];
	        	try {
					smallerLogProbs = getPredsFromBayesNet(currTestInstances);
				} catch (IOException e) {
					e.printStackTrace();
				}
        		
        		//now add predictions from smaller set to currentLogProbs
        		for (int i = 0; i < currSubdivisionSize; i++)
        			currentLogProbs[numInstancesCompleted+i] = smallerLogProbs[i];

	        	numInstancesCompleted += currSubdivisionSize;
        	}
        }
	        
        // Check for NaNs
        for (int j = 0; j < currentLogProbs.length; j++) {
        	if (Double.isNaN(currentLogProbs[j])) {
        		Instance inst = testInstances.instance(j);
        		System.err.println("Error (probably a missing value): got NaN for probability density of instance " + inst);
        	}
        	
        	if (printInfinityDebugInfo
            		&& Double.isInfinite(currentLogProbs[j]))
            	System.err.println("Infinity at item " + j +
            			", testInstance is " + testInstances.instance(j));	    
        }
        
        return currentLogProbs;
	}

	private double[] getPredsFromBayesNet(Instances testInstances) throws IOException{
//    	System.out.println("Learning bayes net!!");
    	//Write positives to a csv file
	    
	    File numericTestingFile = File.createTempFile("numericTesting", ".csv");
	    File factorTestingFile = File.createTempFile("factorTesting", ".csv");
    	 
    	 //Write test numeric data to a csv file
		writeAttrsToFile(testInstances, numericTestingFile, numericAttrs);
    	 
     	 //Write test factor data to a csv file
		writeAttrsToFile(testInstances, factorTestingFile, factorAttrs);

		RCaller caller = Parameters.rCaller;
		RCode code = Parameters.rCode;

		double[] numericResult=null, factorResult=null;
         if (hasNumeric){
        	 String varNames = "c(";
        	 for (int i = 0; i < numNumeric; i++){
        		 if (i != 0) varNames += ", ";
        		 varNames += "\"V" + (i+1)+"\"";
        	 }
        	 varNames += ")";

			 code.clearOnline();

			 code.addRCode("library(bnlearn)");
             code.addRCode("numericTesting = read.csv('" + numericTestingFile.getAbsolutePath() + "', header=FALSE, colClasses=\"numeric\")");

			 if (learnedBayesNetFileNameN == "") {
				 System.out.println("relearning");
				 code.addRCode("numericTraining = read.csv('" + numericTrainingFile.getAbsolutePath() + "', header=FALSE, colClasses=\"numeric\")");
				 if (structurelessBayesNet)
					 code.addRCode("res = empty.graph(" + varNames + ")");
				 else
					 code.addRCode("res = mmhc(numericTraining)");
				 code.addRCode("fit = bn.fit(res, numericTraining)");
			 } else {
				 code.addRCode("load(\"" + learnedBayesNetFileNameN + "\")");
			 }
             
             code.addRCode("result = rep(0," + testInstances.numInstances() + ")");
             code.addRCode("for (i in 1:" + testInstances.numInstances() + ")");
             code.addRCode("result[i] = logLik(fit, numericTesting[i,])");
             
             caller.runAndReturnResultOnline("result");
             numericResult = caller.getParser().getAsDoubleArray("result");
         }

		 if (hasFactor){
        	 String varNames = "c(";
        	 for (int i = 0; i < numFactor; i++){
        		 if (i != 0) varNames += ", ";
        		 varNames += "\"V" + (i+1)+"\"";
        	 }
        	 varNames += ")";

			 code.clearOnline();
			 code.addRCode("library(bnlearn)");

			 code.addRCode("factorTesting = read.csv('" + factorTestingFile.getAbsolutePath() + "', header=FALSE, colClasses=\"factor\")");
			 code.addRCode("factorTraining = read.csv('" + factorTrainingFile.getAbsolutePath() + "', header=FALSE, colClasses=\"factor\")");
			 //make sure levels of testData match levels of training data
			 code.addRCode("for (i in 1:" + numFactor + "){");
			 code.addRCode("levels(factorTesting[,i]) <- union(levels(factorTesting[,i]), levels(factorTraining[,i]))");
			 code.addRCode("levels(factorTraining[,i]) <- levels(factorTesting[,i])}");

			 if (structurelessBayesNet)
				 code.addRCode("res = empty.graph(" + varNames + ")");
			 else
				 code.addRCode("res = mmhc(factorTraining)");

			 if (!useSmoothing)
				 code.addRCode("fit = bn.fit(res, factorTraining)");
			 else {
				 code.addRCode("fitTemp = bn.fit(res, factorTraining)");
				 //now we need to smooth to avoid 0s
				 code.addRCode("source(\"./R-code/smooth.R\")");
				 code.addRCode("fit = bn.smooth(net=fitTemp)");
			 }

			 code.addRCode("result = rep(0," + testInstances.numInstances() + ")");
			 code.addRCode("for (i in 1:" + testInstances.numInstances() + ")");
			 code.addRCode("result[i] = logLik(fit, factorTesting[i,])");

			 caller.runAndReturnResultOnline("result");
			 factorResult = caller.getParser().getAsDoubleArray("result");
		 }

		numericTestingFile.delete();
		factorTestingFile.delete();

         //combine results, if necessary
         double[] result = new double[testInstances.numInstances()];
         if (!hasFactor)
        	 result = numericResult;
         else if (!hasNumeric)
        	 result = factorResult;
         else {
        	 for (int i = 0; i < factorResult.length; i++)
        		 result[i] = factorResult[i]+numericResult[i];
         }

    	return result;
	}
	
	public String getName(){
		return "Bayes Net";
	}

	public void doneWithProbDensity() {
		numericTrainingFile.delete();
		factorTrainingFile.delete();
		new File(learnedBayesNetFileNameN).delete();
	}

}
