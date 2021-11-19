package cade.experiments;

/**
 * Stuff moved from driver relating to UCI data: code that handles data sets, class divisions, etc.
 */
public class UCIExperiments {

	/* (Out-of-date instructions)
	 * if running UCI data on the servers, args are:
	 * args[0]: Which dataset to use. Values: 0 to 15
	 * args[1]: Which class value is the single class for class division.
	 *      Values: 0 to (respectively) 1, 2, 1, 1, 3, 2, 1, 9, 6, 8(* since we never run ERL alone)
	 *      But note: where there are exactly 2 classes, either run 0 and 1 (true), or 0 (true and false).
	 *      Any more would be duplicates.
	 * args[2]: Whether that class value is the single positive class (true) or single negative class (false)
	 * args[3] (optional): directory to store the output in (avg-var lines will be saved to a file)
	 */

	public static void runUCIDataCrossValFromCommandLine(String[] args) {
		if (args.length >= 4)
			System.out.println("Command line args: " + args[0] + " " + args[1] + " " + args[2] + " " + args[3]);
		else
			System.out.println("Command line args: " + args[0] + " " + args[1] + " " + args[2]);
		System.out.println("Running from working Directory " + System.getProperty("user.dir"));

		String[] datasetList = {"adult", "ann-thyroid", "bands", "breast-cancer-wisc",
				"contraceptive-method-choice", "credit", "ecoli", "glass",
				"hayes-roth", "letter", "optdigits", "pendigits", "segment", "yeast", "musk", "ionosphere"};
		String[][] classesPerDataset = {{" >50K", " <=50K"}, {"1", "2", "3"}, {"band", "noband"}, {"2", "4"},
				{"1", "2", "3"}, {"+", "-"}, {"cp", "im", "pp", "imU", "om", "omL", "imL", "imS"}, {"1", "2", "3", "5", "6", "7"}, {"1", "2", "3"}, {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"}, {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"},
				{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}, {"BRICKFACE", "SKY", "FOLIAGE", "CEMENT", "WINDOW", "PATH", "GRASS"}, {"MIT", "NUC", "CYT", "ME1", "ME2", "ME3", "EXC", "VAC", "POX", "ERL"},
				{"0", "1"}, {"g", "b"}};

		int datasetNum = Integer.parseInt(args[0]);
		String dataset = datasetList[datasetNum];

		String filename = "../data/UCIDataSetsCrossVal/" + dataset + ".arff";

//    	String classVar = args[1];
		int classVarNum = Integer.parseInt(args[1]);
		if (classesPerDataset[datasetNum].length < classVarNum + 1) {
			System.out.println("There is no class div " + classVarNum + " on data set " + dataset);
			return;
		}
		String classVar = classesPerDataset[datasetNum][classVarNum];
		boolean isPosClass = Boolean.parseBoolean(args[2]);

		String[] posClasses = new String[isPosClass ? 1 : (classesPerDataset[datasetNum].length - 1)];
		String[] negClasses = new String[classesPerDataset[datasetNum].length - posClasses.length];

		if (isPosClass) {
			posClasses[0] = classVar;
			int count = 0;
			for (int i = 0; i < classesPerDataset[datasetNum].length; i++) {
				if (classesPerDataset[datasetNum][i].equals(classVar))
					continue;
				negClasses[count] = classesPerDataset[datasetNum][i];
				count++;
			}
		} else {
			negClasses[0] = classVar;
			int count = 0;
			for (int i = 0; i < classesPerDataset[datasetNum].length; i++) {
				if (classesPerDataset[datasetNum][i].equals(classVar))
					continue;
				posClasses[count] = classesPerDataset[datasetNum][i];
				count++;
			}
		}

		String resultsFile = null;
		if (args.length >= 4) {
			resultsFile = args[3] + "/" + dataset + "_" + args[1] + (isPosClass ? "Pos" : "Neg") + ".txt";
			System.out.println("Avg-var results will also be in " + resultsFile);
		}

		System.out.print("positive: ");
		for (int j = 0; j < posClasses.length; j++)
			System.out.print(posClasses[j] + " ");
		System.out.print("negative: ");
		for (int j = 0; j < negClasses.length; j++)
			System.out.print(negClasses[j] + " ");
		System.out.println();
		try {
			DriverMethodsLabeledData.runCrossValWithLabeledData(filename, -1, posClasses, negClasses, new int[]{}, 20000, false, resultsFile);
//			System.out.println("running true pos vs. neg baseline");
//			PseudoAnomalyGo.runCrossValWithLabeledData(filename, -1, posClasses, negClasses, new int[]{}, true, resultsFile);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/*
	 * Note -- even though these are the "unsupervised" data sets, they get run identically to the others.
	 * Why? --> I think we have to manually change the settingParams within runCrossValWithLabeledData() to 21, so it
	 * triggers using a different class (UnsupervisedArffDataGenerator) for data generation.
	 * todo: See also PseudoAnomalyGo.runUnlabeledWithCommandLineArgs()
	 *
	 * args[0]: Which dataset to use. Values: 0-5 (normalized Vegas), 6-11 (raw Vegas), 12 coil, 13 shuttle
	 * args[1]: Which class is the outlier class. Any of 2,3,5,6,7 for shuttle; for Vegas & coil, will be ignored, so use 0.
	 * args[2] (optional): directory to store the output in (avg-var lines will be saved to a file)
	 */
	public static void runUCIDataUnsupFromCommandLine(String[] args) {
		if (args.length >= 3)
			System.out.println("Command line args: " + args[0] + " " + args[1] + " " + args[2]);
		else
			System.out.println("Command line args: " + args[0] + " " + args[1]);
		System.out.println("Running from working Directory " + System.getProperty("user.dir"));

		String[] datasetList = {"sep-2012-up", "oct-2012-up", "nov-2012-up", "dec-2012-up",
				"jan-2013-up", "feb-2013-up",
				"sep-2012", "oct-2012", "nov-2012", "dec-2012", "jan-2013", "feb-2013",
				"coil", "shuttle"};

		int datasetNum = Integer.parseInt(args[0]);
		String dataset = datasetList[datasetNum];
//		String filename = "../data/unsupervisedDataSets/" + dataset + ".arff";
		String filename = "/Users/lfriedl/Documents/lab-work/pseudo-anomaly/data/unsupervisedDataSets/" + dataset + ".arff";

		// For Vegas or coil
		String[] posClasses = new String[]{"0"};
		String[] negClasses = new String[]{"1"};

		// For shuttle
		if (datasetNum == 13) {
			posClasses = new String[]{"1"};
			negClasses = new String[]{args[1]};
			int outlierClassNum = Integer.parseInt(args[1]);
			if (outlierClassNum != 2 && outlierClassNum != 3 && outlierClassNum != 5 &&
					outlierClassNum != 6 && outlierClassNum != 7) {
				System.err.println("For shuttle data, outlier class must be one of 2,3,5,6,7");
				return;
			}
		}

		String resultsFile;
		if (args.length > 3) {
			resultsFile = args[2] + "/" + dataset + "_" + args[1] + "_" + args[3] + ".txt";
		} else {
			resultsFile = args[2] + "/" + dataset + "_" + args[1] + ".txt";
		}
		System.out.println("Avg-var results will also be in " + resultsFile);

		System.out.print("positive: ");
		for (int j = 0; j < posClasses.length; j++)
			System.out.print(posClasses[j] + " ");
		System.out.print("negative: ");
		for (int j = 0; j < negClasses.length; j++)
			System.out.print(negClasses[j] + " ");
		System.out.println();

		try {
			DriverMethodsLabeledData.runCrossValWithLabeledData(filename, -1, posClasses, negClasses, new int[]{}, -1, false, resultsFile);
			//System.out.println("running true pos vs. neg baseline");
			//PseudoAnomalyGo.runCrossValWithLabeledData(filename, -1, posClasses, negClasses, new int[]{}, true, resultsFile);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}