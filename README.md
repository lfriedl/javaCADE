**Dependencies**

JavaCADE depends on some external packages. Their jar files are expected to be found in the subdirectory `lib/` (if compiling with `ant`) or elsewhere in the classpath.
* [Weka](https://waikato.github.io/weka-wiki/) machine learning library, version 3.8.5. (Used for its classifiers, namely Random Forest and KNN.) 
    * Follow the [download instructions](https://waikato.github.io/weka-wiki/downloading_weka/) for version 3.8 to get the file `weka.jar`. Then, to avoid an [error loading weka](https://stackoverflow.com/questions/42178995/weka-linear-regression-classnotfoundexception), go to the directory where that jar is located and extract 3 files from it by running this command: `jar xf weka.jar arpack_combined.jar core.jar mtj.jar`. Finally, put all 4 jar files into the classpath (`ant` expects them in the subdirectory `lib/weka-3-8-5/`).
    * One of Weka's user-contributed packages, [localOutlierFactor v.1.4.0](https://weka.sourceforge.io/packageMetaData/localOutlierFactor/1.0.4.html) (must be exactly that version). Download the bundle from that link, then put the resulting `localOutlierFactor.jar` into the classpath.   
* [RCaller](https://github.com/jbytecode/rcaller) version ≥4.0.0, which lets Java execute scripts in the R language. It (plus its dependencies) can be obtained using Maven (ID: `com.github.jbytecode`) or by downloading this jar file: [RCaller-4.0.2-jar-with-dependencies.jar](https://github.com/jbytecode/rcaller/releases/download/RCaller-4.0.2/RCaller-4.0.2-jar-with-dependencies.jar). (JavaCADE relies on R for most of the probability density estimates, apart from UNIFORM--that is, for MARGINAL_KDE, BAYESNET, and GAUSSIAN.)
    * For RCaller to work, you must also have [R installed](https://www.r-project.org/). I am using version 3.6.3.  
    * Within R, four packages and their dependencies need to be installed (otherwise, errors will be thrown at runtime). At the R prompt, run: `install.packages(c("ks", "KernSmooth", "mnormt", "bnlearn"))`. For reference, I'm using the following versions: ks (1.13.2), KernSmooth (2.23.20), mnormt (2.0.2), and bnlearn (4.7). (The version number can be checked using `packageVersion("ks")`, and previous versions can be installed using [the remotes package](https://stackoverflow.com/a/29840882/).)
      


**Building**

A compiled .jar file will be provided, so there's no need to compile & build the project unless you want to modify the source code.
The project can be built from the command line or in an IDE.

To build at the command line, simply run `ant` from this directory. It will use the file `build.xml`.


Notes: 
* I've configured this project using openjdk-17 as the language SDK, with language level 16. (I believe any JDK version 11 or higher would work.)
* Requires [ant](http://ant.apache.org/) version ≥1.8.0.  
* Excludes unit tests by default. To include them, run this:
`ant -Dskip.tests=false`.
* Building and running the tests requires an additional dependency, the Junit 4 library. (My IDE resolved this by placing two files in the lib directory: `junit-4.13.1.jar` and `hamcrest-core-1.3.jar`).


**Running**