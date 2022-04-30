##CADE

This is the code base used for the paper "[Classifier-Adjusted Density Estimation for Anomaly Detection and One-Class Classification](http://dx.doi.org/10.1137/1.9781611973440.67)." (By Lisa Friedland, Amanda Gentzel, and David Jensen. In <i>Proceedings of the 2014 SIAM International Conference on Data Mining (SDM)</i>, pp. 578&ndash;586.)

There's a (rarely seen, but useful) [supplementary material](https://lfriedl.github.io/pubs/SDM2014-supp.pdf) document that discusses some of the implementation details. 

###What is it?

Briefly, CADE is a technique to estimate the probability density function of a data set. This is a useful way of doing anomaly detection (low values = anomalies).

We didn't invent it (it's described in the textbook [ESL](https://hastie.su.domains/ElemStatLearn)), but we tried to draw attention to how robustly such a "simple" method works, especially for high-dimensional data (e.g., >10-20 attributes).

The intuition behind the method is somewhat similar to that behind [GANs](https://en.wikipedia.org/wiki/Generative_adversarial_network) (which came out later). First 
you make an initial (naive) approximation of the data. Then you train a classifier to distinguish between the true data and your approximation. 
You use the classifier's predictions to update the initial approximation (here, via Bayes' rule). With CADE, you do this update just once and stop.

Part of the "simplicity" of CADE is that it relies on standard, well-studied tools. 
We found good performance using (for example) a classifier of Random Forest, combined with 
a density estimator that learns a separate 1-d kernel density estimator for each attribute and 
combines them as if the attributes were independent.

###Ugh, bug in LOF

Not our fault! However... 

Weka's implementation of Local Outlier Factor (LOF) had a few bugs. We used this as a competitor method.

_(Note: we did notice and fix some of those bugs at the time.)_

Running today's corrected version (in 2021, while preparing this package for release), it looks as though the results with LOF itself are unchanged.

However, running Bagged LOF (using the same wrapper code as before) now gives _better_ results than before. 
In other words, in the paper, the performance of Bagged LOF was artificially low. 

This means all the paper's comparisons of CADE to Bagged LOF need to be rerun/revisited. 
(Alas, no good deed, such as releasing potentially useful code from an old project, goes unpunished.)

##Code and Usage

###Basic inputs and outputs

###Code base

The heart of CADE is found in the method `runClassifierAndPrintPredictions()` inside the main driver file, `PseudoAnomalyGo.java`.

When this is called, it will use: 

* Information stored in `Parameters` and `ParamsMultiRuns` (some of which you'll have provided at the command line).
* One of Weka's classifiers, from `cade.classifiers.LocalClassifier`.
* A generative probability density object from `cade.generativeProbDensities`, which in turn will hold a corresponding estimator from `cade.estimators`. 
* A `FormulaDensityCombiner` object, which does the actual math (the Bayes' rule calculation) in a method called `scoreForInstance()`. 
* The appropriate object (here, a `NoLabelsArffDataGenerator`) from `cade.positiveDataGenerators`. These are the classes that understand what format the data comes in (e.g., unlabeled, partly labeled, fully labeled).

N.B.
There are a lot of source files, for two reasons:
1. In Java, this is pretty inevitable.
2. Most of the code is for handling experiments: running CADE on a single dataset (with known ground truth) in numerous ways, while storing and printing the parameters, the results, and other stats and diagnostics.

In cleaning up the code for release, I prioritized:
1. Having a useful default behavior at the command line. (It takes the input data and appends a column of scores.)
2. Making sure the main cross-validated experiments still replicate.
3. Releasing most of the other code that we relied upon. (Caveat: some functionality may have been broken during refactoring.)

##Dependencies

JavaCADE depends on some external packages. It expects to find their jar files in the subdirectory `lib/` (if compiling with `ant`) or elsewhere in the classpath.

* [Weka](https://waikato.github.io/weka-wiki/) machine learning library, version 3.8.5. (Used for the classifiers, e.g., Random Forest and KNN.) 
    * Obtain the file `weka.jar`, either by [downloading this zip bundle](https://sourceforge.net/projects/weka/files/weka-3-8/3.8.5/weka-3-8-5.zip/) or by following your system's [installation instructions](https://waikato.github.io/weka-wiki/downloading_weka/) for version 3.8.
      Then, to avoid an [error loading weka](https://stackoverflow.com/questions/42178995/weka-linear-regression-classnotfoundexception),
      go to the directory where that jar is located and extract 3 files from it by running this command: 
      `jar xf weka.jar arpack_combined.jar core.jar mtj.jar`.
    * Finally, put all 4 jar files into the classpath. Put them into the subdirectory `lib/weka-3-8-5/` to make all the commands below work.
    * One of Weka's user-contributed packages, [localOutlierFactor v.1.4.0](https://weka.sourceforge.io/packageMetaData/localOutlierFactor/1.0.4.html). (Must be exactly that version. Earlier versions had bugs, and later versions changed the interface.) Download the bundle from that link, then put the resulting `localOutlierFactor.jar` into the classpath.   
* [RCaller](https://github.com/jbytecode/rcaller) version ≥4.0.0, which lets Java execute scripts in the R language. It (plus its dependencies) can be obtained by downloading [this jar file](https://github.com/jbytecode/rcaller/releases/download/RCaller-4.0.2/RCaller-4.0.2-jar-with-dependencies.jar), or alternatively, by using Maven (ID: `com.github.jbytecode`). (JavaCADE calls R for most of the probability density estimates, apart from UNIFORM--that is, for MARGINAL_KDE, BAYESNET, and GAUSSIAN.)


* For RCaller to work, you must also have [R installed](https://www.r-project.org/). I am using version 3.6.3.  
  * Within R, four packages and their dependencies need to be installed (otherwise, errors will be thrown at runtime). 
  At the R prompt, run: `install.packages(c("ks", "KernSmooth", "mnormt", "bnlearn"))`. For reference, I'm using the following versions: ks (1.13.2), KernSmooth (2.23.20), mnormt (2.0.2), and bnlearn (4.7). (The version number can be checked using `packageVersion("ks")`, and previous versions can be installed using [the remotes package](https://stackoverflow.com/a/29840882/).)

To summarize, here's what my `lib/` contains:
* `weka-3-8-5/` has `arpack_combined.jar core.jar mtj.jar weka.jar`
* `localOutlierFactor.jar`
* `RCaller-4.0.2-jar-with-dependencies.jar`

##Building

A compiled .jar file will (eventually) be provided, so there will be no need to compile & build the project unless you want to modify the source code. 

The project can be built from the command line or in an IDE.

To build at the command line, simply run `ant` from this directory. It will use the file `build.xml`.

Notes for command line:

* The project can be built (`javac`) using Java (JDK) versions 11 or higher. It is configured to compile into bytecode version 11. (This means it can be run with `java` in that version or higher.) 
To make `ant` use a specific JDK, set it in the environment variable JAVA_HOME.
* Requires [ant](http://ant.apache.org/) version ≥1.8.0.  
* Unit tests are not (currently) included in git. If/when they are added:
  * The build file excludes unit tests by default. To include them, run this:
`ant -Dskip.tests=false`. 
  * Building and running the tests requires an additional dependency, the Junit 4 library. (My IDE resolved this by placing two files in the lib directory: `junit-4.13.1.jar` and `hamcrest-core-1.3.jar`).


##Running

###Using the compiled files.

Run it using the files compiled to `out/production`:

`java -classpath out/production/CADE:lib/RCaller-4.0.2-jar-with-dependencies.jar:lib/weka-3-8-5/arpack_combined.jar:lib/weka-3-8-5/mtj.jar:lib/weka-3-8-5/core.jar:lib/weka-3-8-5/weka.jar -Xmx1g cade.PseudoAnomalyGo <args-to-the-program>`
* `java` (must be version 11 or higher).
* The classpath includes the location of the compiled javaCade project, plus the 5 dependency jar files.
* `cade.PseudoAnomalyGo` is the main class to call.
* `-Xmx1g` increases the memory.
* `<args-to-the-program>` will look something like, for example:
  `../data/UCIDataSetsRevised/coil.arff ../outputs/coil.csv 0 0 -1 85`.
  Run the program without arguments to print the documentation describing them.


###Using the jar file.

Or use CADE.jar. The following syntax works if CADE.jar, plus all 
the Weka and RCaller .jar files, is found in the directory `out/artifacts/CADE_jar/`.

`java -jar out/artifacts/CADE_jar/CADE.jar <args-to-the-program>` 

Or equivalently,

`java -cp "out/artifacts/CADE_jar/*" cade.PseudoAnomalyGo <args-to-the-program>` 

