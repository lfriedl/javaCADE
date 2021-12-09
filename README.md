**Dependencies**

JavaCADE depends on some external packages. Their jar files are expected to be found in the subdirectory `lib/` (if compiling with `ant`) or elsewhere in the classpath.
* [Weka](https://waikato.github.io/weka-wiki/) machine learning library, version 3.8.5. (Used for its classifiers, e.g., Random Forest and KNN.) 
    * Obtain the file `weka.jar`, either by [downloading this zip bundle](https://sourceforge.net/projects/weka/files/weka-3-8/3.8.5/weka-3-8-5.zip/) or by following your system's [installation instructions](https://waikato.github.io/weka-wiki/downloading_weka/) for version 3.8.
      Then, to avoid an [error loading weka](https://stackoverflow.com/questions/42178995/weka-linear-regression-classnotfoundexception),
      go to the directory where that jar is located and extract 3 files from it by running this command: 
      `jar xf weka.jar arpack_combined.jar core.jar mtj.jar`. 
      Finally, put all 4 jar files into the classpath (`ant` expects them in the subdirectory `lib/weka-3-8-5/`).
    * (Optional! Only needed if using LocalOutlierFactor.) One of Weka's user-contributed packages, [localOutlierFactor v.1.4.0](https://weka.sourceforge.io/packageMetaData/localOutlierFactor/1.0.4.html) (must be exactly that version). Download the bundle from that link, then put the resulting `localOutlierFactor.jar` into the classpath.   
* [RCaller](https://github.com/jbytecode/rcaller) version ≥4.0.0, which lets Java execute scripts in the R language. It (plus its dependencies) can be obtained by downloading [this jar file](https://github.com/jbytecode/rcaller/releases/download/RCaller-4.0.2/RCaller-4.0.2-jar-with-dependencies.jar) or by using Maven (ID: `com.github.jbytecode`). (JavaCADE calls R for most of the probability density estimates, apart from UNIFORM--that is, for MARGINAL_KDE, BAYESNET, and GAUSSIAN.)


* For RCaller to work, you must also have [R installed](https://www.r-project.org/). I am using version 3.6.3.  
  * Within R, four packages and their dependencies need to be installed (otherwise, errors will be thrown at runtime). At the R prompt, run: `install.packages(c("ks", "KernSmooth", "mnormt", "bnlearn"))`. For reference, I'm using the following versions: ks (1.13.2), KernSmooth (2.23.20), mnormt (2.0.2), and bnlearn (4.7). (The version number can be checked using `packageVersion("ks")`, and previous versions can be installed using [the remotes package](https://stackoverflow.com/a/29840882/).)



**Building**

A compiled .jar file will (eventually) be provided, so there will be no need to compile & build the project unless you want to modify the source code. 

The project can be built from the command line or in an IDE.

To build at the command line, simply run `ant` from this directory. It will use the file `build.xml`.

Notes for command line:

* The project can be built using Java (JDK) versions 11 or higher, and it is configured to compile to bytecode version 11 (so can be run with `java` in that version or higher). To make `ant` use a specific JDK, set it in the environment variable JAVA_HOME.
* Requires [ant](http://ant.apache.org/) version ≥1.8.0.  
* Unit tests are not (yet) included in git. If/when they are added:
  * The build file excludes unit tests by default. To include them, run this:
`ant -Dskip.tests=false`. 
  * Building and running the tests requires an additional dependency, the Junit 4 library. (My IDE resolved this by placing two files in the lib directory: `junit-4.13.1.jar` and `hamcrest-core-1.3.jar`).


**Running**

Using the jar:

`java -jar out/artifacts/CADE_jar/CADE.jar <args-to-the-program>` 

Or equivalently,

`java -cp "out/artifacts/CADE_jar/*" cade.PseudoAnomalyGo <args-to-the-program>` 


* This assumes the directory `out/artifacts/CADE_jar` is the one containing CADE.jar along with the Weka and RCaller jar files.

* `<args-to-the-program>` will look something like, for example:
`../data/UCIDataSetsRevised/coil.arff ../outputs/coil.csv 0 0 -1 85`. 
Run the program without arguments to print the documentation describing them. 

Or run it without a jar, instead using the files compiled to `out/production`:

`java -classpath out/production/CADE:lib/RCaller-4.0.2-jar-with-dependencies.jar:lib/weka-3-8-5/arpack_combined.jar:lib/weka-3-8-5/mtj.jar:lib/weka-3-8-5/core.jar:lib/weka-3-8-5/weka.jar -Xmx1g cade.PseudoAnomalyGo <args-to-the-program>`
* `java` (must be version 11 or higher).
* The classpath includes the location of the compiled javaCade project, plus the 5 dependency jar files.
* `cade.PseudoAnomalyGo` is the main class to call.
* `-Xmx1g` increases the memory.
