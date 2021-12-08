package cade.positiveDataGenerators;

import java.io.IOException;
import java.util.Random;

import cade.Parameters;
import com.github.rcaller.rstuff.RCaller;
import com.github.rcaller.rstuff.RCode;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

abstract public class LabeledArffDataGenerator extends DatafileDataGenerator {

    int numInstancesUsedInDataSet; //number of data instances used from file = min(allData.numInstances, maxNumInstances wanted)
    boolean normalize;
    int classAttr;
    double avgAbsCorr = 0;
    boolean calculateAvgAbsCorr = false;
    boolean addNoiseAttrs = false;
    int numNoiseAttrs = 0;
    int[] nonsenseRange = new int[]{0,16};

    public void initialize() throws IOException {
        loadData();

        if (classAttr == -1)
            classAttr = allData.numAttributes() - 1;
        allData.setClassIndex(classAttr);
        allData.randomize(randomness);

        allData = setUpClassLabel(allData); // (replaces any previous classIndex)

        removeUnwantedAttributes(allData, attributesToUse);
        
        if (addNoiseAttrs)
        	addNoiseAttributes();
        
        numAttributes = allData.numAttributes() - 1;

        if (normalize)
        	normalize();

    }

    // Replace old classIndex with one we construct here.
    // Using variables positiveClasses and negativeClasses, creates new class label with classes "positive" and
    // "negative", as final attribute. Deletes instances if their class label isn't in either list.
    protected Instances setUpClassLabel(Instances data) {
        // Add a new attribute to hold the class label
        FastVector classVals = new FastVector(2);
        classVals.addElement(Parameters.positiveClassLabel);
        classVals.addElement(Parameters.negativeClassLabel);

        int oldIndex = data.classAttribute().index();
        Attribute classAttr = new Attribute("classAttr", classVals);
        data.insertAttributeAt(classAttr, data.numAttributes());

        Instances newData = new Instances(data, data.numInstances());

        for (int i = 0; i < data.numInstances(); i++) {
            Instance inst = data.instance(i);
            if (valueInClass(inst.classValue(), positiveClasses, data.firstInstance())) {//if positive
                inst.setValue(data.numAttributes() - 1, Parameters.positiveClassLabel);
                newData.add(inst);
            } else if (valueInClass(inst.classValue(), negativeClasses, data.firstInstance())) {//if negative
                inst.setValue(data.numAttributes() - 1, Parameters.negativeClassLabel);
                newData.add(inst);
            }
        }
        newData.setClassIndex(data.numAttributes() - 1);
        newData.deleteAttributeAt(oldIndex);
        return newData;
    }

        //normalizes allData (sets all continuous values to be (val-mean)/stdev)
    private void normalize(){
    	for (int i = 0; i < allData.numAttributes(); i++){
    		if (!allData.attribute(i).isNominal()){ //not nominal means it needs normalized
    			//first, loop over all instances to get the mean and stdDev
	    		double mean = 0;
	    		double stdDev = 0;
	    		double sumOfSquares = 0;
	    		for (int j = 0; j < allData.numInstances(); j++){
	    			mean += allData.instance(j).value(i);
	    			sumOfSquares += Math.pow(allData.instance(j).value(i), 2);
	    		}
	    		mean /= allData.numInstances();
	    		stdDev = Math.sqrt((sumOfSquares/allData.numInstances())-(mean*mean));

	    		//now, loop over all instances and change values to normalized ones
	    		for (int j = 0; j < allData.numInstances(); j++)
	    			allData.instance(j).setValue(i, (allData.instance(j).value(i)-mean)/stdDev);
    		}
    	}
    }

    // Calculates average absolute value of correlation for everything in the positive class.
    // Only works (I think) for numeric-only data.
    protected void calculateCorrelation() {
        //calculate average absolute value of correlation
        if (calculateAvgAbsCorr){
	        avgAbsCorr = 0;
            RCaller caller = RCaller.create();
//            caller.cleanRCode();
	        RCode code = RCode.create();

	        //first, write all the positive data to get the correlation matrix
	        Instance oneInst = null;
	        boolean valWritten = false;
	        for (int i = 0; i < allData.numInstances(); i++){
                Instance inst = allData.instance(i);
                if (!allData.classAttribute().value((int) inst.classValue()).equals(Parameters.positiveClassLabel))
                    continue;

                String val = "";
                if (!valWritten){
                    val = "data = c(";
                    for (int k = 0; k < inst.numAttributes()-1; k++){
                        if (k != 0)
                            val += ",";
                        val += inst.value(k);
                    }
                    val += ")";
                    valWritten = true;
                    oneInst = inst;
                } else {
                    val = "data = rbind(data, c(";
                    for (int k = 0; k < inst.numAttributes()-1; k++){
                        if (k != 0)
                            val += ",";
                        val += inst.value(k);
                    }
                    val += "))";
                }
                code.addRCode(val);
//	            System.out.println(val);
//	        	}
	        }

	        code.addRCode("corr = cor(data)");
	        code.addRCode("for (i in 1:length(corr[1,])){");
	        code.addRCode("for (j in 1:length(corr[1,])){");
	        code.addRCode("if (is.na(corr[i,j]))");
	        code.addRCode("corr[i,j] = -1}}");
	        caller.setRCode(code);
	        caller.runAndReturnResult("corr");

	        double[][] corrMat = caller.getParser().getAsDoubleMatrix("corr", oneInst.numAttributes()-1, oneInst.numAttributes()-1);
	        int count = 0;
	        for (int x = 0; x < numAttributes; x++){
        		for (int y = x+1; y < numAttributes; y++){
        			if (corrMat[x][y] != -1){
        				avgAbsCorr += Math.abs(corrMat[x][y]);
        				count++;
        			}
        		}
        	}
        	avgAbsCorr /= count;
	        System.out.println("avgAbsCorr = " + avgAbsCorr);
	        
//	        files[0].delete();
//	        files[1].delete();
        }
    }

    private void addNoiseAttributes(){
    	Random rng = new Random();
    	int currNumAttrs = allData.numAttributes();
    	
    	if (numNoiseAttrs == -1)
    		numNoiseAttrs = currNumAttrs - 1;
    	//Now, if we need nonsenseDims, add those in
    	for (int i = 0; i < numNoiseAttrs; i++){
    		allData.insertAttributeAt(new Attribute("Nonsense" + (i+1)), currNumAttrs-1+i);
    		for (int j = 0; j < allData.numInstances(); j++){
    			allData.instance(j).setValue(currNumAttrs-1+i, rng.nextInt(1+nonsenseRange[1]-nonsenseRange[0])+nonsenseRange[0]);
    		}
    	}
    }
}
