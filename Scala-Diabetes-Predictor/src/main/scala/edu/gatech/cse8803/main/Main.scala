/**
 * @author Tian Tan <ttan40@gatech.edu>.
 */

package edu.gatech.cse8803.main

import java.text.SimpleDateFormat

import edu.gatech.cse8803.clustering.{Metrics, NMF}
import edu.gatech.cse8803.features.FeatureConstruction
import edu.gatech.cse8803.ioutils.CSVUtils
import edu.gatech.cse8803.model.{Diagnostic, LabResult, Medication}
import edu.gatech.cse8803.phenotyping.T2dmPhenotype
import org.apache.spark.mllib.clustering.{GaussianMixture, KMeans}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrices, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

import scala.io.Source


object Main {
  def main(args: Array[String]) {
    import org.apache.log4j.{Level, Logger}

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val sc = createContext
    val sqlContext = new SQLContext(sc)

    /** initialize loading of data */
    val (medication, labResult, diagnostic) = loadRddRawData(sqlContext)
    val (candidateMedication, candidateLab, candidateDiagnostic) = loadLocalRawData


    /** conduct phenotyping */
    val phenotypeLabel = T2dmPhenotype.transform(medication, labResult, diagnostic)

    /** feature construction with all features */
    val featureTuples = sc.union(
      FeatureConstruction.constructDiagnosticFeatureTuple(diagnostic),
      FeatureConstruction.constructLabFeatureTuple(labResult),
      FeatureConstruction.constructMedicationFeatureTuple(medication)
    )

    val rawFeatures = FeatureConstruction.construct(sc, featureTuples)

    val (kMeansPurity, gaussianMixturePurity, nmfPurity) = testClustering(phenotypeLabel, rawFeatures)
    println(f"[All feature] purity of kMeans is: $kMeansPurity%.5f")
    println(f"[All feature] purity of GMM is: $gaussianMixturePurity%.5f")
    println(f"[All feature] purity of NMF is: $nmfPurity%.5f")

    /** feature construction with filtered features */
    val filteredFeatureTuples = sc.union(
      FeatureConstruction.constructDiagnosticFeatureTuple(diagnostic, candidateDiagnostic),
      FeatureConstruction.constructLabFeatureTuple(labResult, candidateLab),
      FeatureConstruction.constructMedicationFeatureTuple(medication, candidateMedication)
    )

    val filteredRawFeatures = FeatureConstruction.construct(sc, filteredFeatureTuples)

    val (kMeansPurity2, gaussianMixturePurity2, nmfPurity2) = testClustering(phenotypeLabel, filteredRawFeatures)
    println(f"[Filtered feature] purity of kMeans is: $kMeansPurity2%.5f")
    println(f"[Filtered feature] purity of GMM is: $gaussianMixturePurity2%.5f")
    println(f"[Filtered feature] purity of NMF is: $nmfPurity2%.5f")
    sc.stop 
  }

  def testClustering(phenotypeLabel: RDD[(String, Int)], rawFeatures:RDD[(String, Vector)]): (Double, Double, Double) = {
    import org.apache.spark.mllib.linalg.Matrix
    import org.apache.spark.mllib.linalg.distributed.RowMatrix

    /** scale features */
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(rawFeatures.map(_._2))
    val features = rawFeatures.map({ case (patientID, featureVector) => (patientID, scaler.transform(Vectors.dense(featureVector.toArray)))})
    val rawFeatureVectors = features.map(_._2).cache()

    /** reduce dimension */
    val mat: RowMatrix = new RowMatrix(rawFeatureVectors)
    val pc: Matrix = mat.computePrincipalComponents(10) // Principal components are stored in a local dense matrix.
    val featureVectors = mat.multiply(pc).rows

    val densePc = Matrices.dense(pc.numRows, pc.numCols, pc.toArray).asInstanceOf[DenseMatrix]
    /** transform a feature into its reduced dimension representation */
    def transform(feature: Vector): Vector = {
      Vectors.dense(Matrices.dense(1, feature.size, feature.toArray).multiply(densePc).toArray)
    }

    /** TODO: K Means Clustering using spark mllib
      *  Train a k means model using the variabe featureVectors as input
      *  Set maxIterations =20 and seed as 8803L
      *  Assign each feature vector to a cluster(predicted Class)
      *  Obtain an RDD[(Int, Int)] of the form (cluster number, RealClass)
      *  Find Purity using that RDD as an input to Metrics.purity
      *  Remove the placeholder below after your implementation
      **/
    val numClusters = 3
    val maxIterations = 20
    val seed = 8803L
    featureVectors.cache()
    val kMeansModel = KMeans.train(featureVectors, numClusters, maxIterations,1,"k-means||",seed)
    val predictionClusterID = kMeansModel.predict(featureVectors)
    val rawFeatureID = features.map(_._1)
    val kmeansAssignment = rawFeatureID.zip(predictionClusterID)
    val compareRDD = kmeansAssignment.join(phenotypeLabel).map(_._2)
    val kMeansPurity = Metrics.purity(compareRDD)


    /* compare clustering results */
   val Case = compareRDD.filter(line=>line._2==1)
   val Casesize = Case.count().toDouble
   val casecluster=Case.map(line=>(line,1)).keyBy(_._1).reduceByKey((x,y)=>(x._1,x._2+y._2)).map(x=>(x._1,(x._2._2.toDouble/Casesize)))
   val Control = compareRDD.filter(line=>line._2==2)
   val Controlsize = Control.count().toDouble
   val controlcluster = Control.map(line=>(line,1)).keyBy(_._1).reduceByKey((x,y)=>(x._1,x._2+y._2)).map(line=>(line._1,line._2._2.toDouble/Controlsize))
   val Other = compareRDD.filter(line=>line._2==3)
   val Othersize = Other.count().toDouble
   val othercluster = Other.map(line=>(line,1)).keyBy(_._1).reduceByKey((x,y)=>(x._1,x._2+y._2)).map(line=>(line._1,line._2._2.toDouble/Othersize))
   println(f"kmeans")
   casecluster.foreach(println)
   controlcluster.foreach(println)
   othercluster.foreach(println)


    /** TODO: GMMM Clustering using spark mllib
      *  Train a Gaussian Mixture model using the variabe featureVectors as input
      *  Set maxIterations =20 and seed as 8803L
      *  Assign each feature vector to a cluster(predicted Class)
      *  Obtain an RDD[(Int, Int)] of the form (cluster number, RealClass)
      *  Find Purity using that RDD as an input to Metrics.purity
      *  Remove the placeholder below after your implementation
      **/
    val GMMmodel = new GaussianMixture().setK(numClusters).setMaxIterations(maxIterations).setSeed(seed).run(featureVectors)
    val predictionGMMID = GMMmodel.predict(featureVectors)
    val GMMassignment = rawFeatureID.zip(predictionGMMID)
    val compareGMM = GMMassignment.join(phenotypeLabel).map(_._2)
    val gaussianMixturePurity = Metrics.purity(compareGMM)


   val CaseGMM = compareGMM.filter(line=>line._2==1)
   val CasesizeGMM = CaseGMM.values.count().toDouble
   val caseclusterGMM=CaseGMM.map(line=>(line,1)).keyBy(_._1).reduceByKey((x,y)=>(x._1,x._2+y._2)).map(x=>(x._1,(x._2._2.toDouble/CasesizeGMM)))
   val ControlGMM = compareGMM.filter(line=>line._2==2)
   val ControlsizeGMM = ControlGMM.count().toDouble
   val controlclusterGMM = ControlGMM.map(line=>(line,1)).keyBy(_._1).reduceByKey((x,y)=>(x._1,x._2+y._2)).map(line=>(line._1,line._2._2.toDouble/ControlsizeGMM))
   val OtherGMM = compareGMM.filter(line=>line._2==3)
   val OthersizeGMM = OtherGMM.count().toDouble
   val otherclusterGMM = OtherGMM.map(line=>(line,1)).keyBy(_._1).reduceByKey((x,y)=>(x._1,x._2+y._2)).map(line=>(line._1,line._2._2.toDouble/OthersizeGMM))
   println(f"GMM")
   caseclusterGMM.foreach(println)
   controlclusterGMM.foreach(println)
   otherclusterGMM.foreach(println)

    /** NMF */
    val rawFeaturesNonnegative = rawFeatures.map({ case (patientID, f)=> Vectors.dense(f.toArray.map(v=>Math.abs(v)))})
    val (w, _) = NMF.run(new RowMatrix(rawFeaturesNonnegative), 3, 100)
    // for each row (patient) in W matrix, the index with the max value should be assigned as its cluster type
    val assignments = w.rows.map(_.toArray.zipWithIndex.maxBy(_._1)._2)
    // zip patientIDs with their corresponding cluster assignments
    // Note that map doesn't change the order of rows
    val assignmentsWithPatientIds=features.map({case (patientId,f)=>patientId}).zip(assignments) 
    // join your cluster assignments and phenotypeLabel on the patientID and obtain a RDD[(Int,Int)]
    // which is a RDD of (clusterNumber, phenotypeLabel) pairs 
    val nmfClusterAssignmentAndLabel = assignmentsWithPatientIds.join(phenotypeLabel).map({case (patientID,value)=>value})
    // Obtain purity value
    val nmfPurity = Metrics.purity(nmfClusterAssignmentAndLabel)

//    val CaseNMF = nmfClusterAssignmentAndLabel.filter(line=>line._2==1)
//    val CasesizeNMF = CaseNMF.values.count().toDouble
//    val caseclusterNMF=CaseNMF.map(line=>(line,1)).keyBy(_._1).reduceByKey((x,y)=>(x._1,x._2+y._2)).map(x=>(x._1,(x._2._2.toDouble/CasesizeNMF)))
//    val ControlNMF = nmfClusterAssignmentAndLabel.filter(line=>line._2==2)
//    val ControlsizeNMF = ControlNMF.count().toDouble
//    val controlclusterNMF = ControlNMF.map(line=>(line,1)).keyBy(_._1).reduceByKey((x,y)=>(x._1,x._2+y._2)).map(line=>(line._1,line._2._2.toDouble/ControlsizeNMF))
//    val OtherNMF = nmfClusterAssignmentAndLabel.filter(line=>line._2==3)
//    val OthersizeNMF = OtherNMF.count().toDouble
//    val otherclusterNMF = OtherNMF.map(line=>(line,1)).keyBy(_._1).reduceByKey((x,y)=>(x._1,x._2+y._2)).map(line=>(line._1,line._2._2.toDouble/OthersizeNMF))
//    println(f"NMF")
//    caseclusterNMF.foreach(println)
//    controlclusterNMF.foreach(println)
//    otherclusterNMF.foreach(println)



    (kMeansPurity, gaussianMixturePurity, nmfPurity)
  }

  /**
   * load the sets of string for filtering of medication
   * lab result and diagnostics
    *
    * @return
   */
  def loadLocalRawData: (Set[String], Set[String], Set[String]) = {
    val candidateMedication = Source.fromFile("data/med_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    val candidateLab = Source.fromFile("data/lab_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    val candidateDiagnostic = Source.fromFile("data/icd9_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    (candidateMedication, candidateLab, candidateDiagnostic)
  }


  def loadRddRawData(sqlContext: SQLContext): (RDD[Medication], RDD[LabResult], RDD[Diagnostic]) = {
    /** You may need to use this date format. */
    val dateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssX")

    /** load data using Spark SQL into three RDDs and return them
      * Hint: You can utilize edu.gatech.cse8803.ioutils.CSVUtils and SQLContext.
      *
      * Notes:Refer to model/models.scala for the shape of Medication, LabResult, Diagnostic data type.
      *       Be careful when you deal with String and numbers in String type.
      *       Ignore lab results with missing (empty or NaN) values when these are read in.
      *       For dates, use Date_Resulted for labResults and Order_Date for medication.
      * */
    /** TODO: implement your own code here and remove existing placeholder code below */
    val medrows = CSVUtils.loadCSVAsTable(sqlContext,"data/medication_orders_INPUT.csv","medicine")
    val medrowsSQL = sqlContext.sql("SELECT Member_ID,Order_Date,Drug_Name FROM medicine")
    /* val format = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
       var date_temp = format.parse("2014-12-21 00:19:58Z")*/
    val medication: RDD[Medication] =  medrowsSQL.map(line=>Medication(line(0).toString,dateFormat.parse(line(1).toString),line(2).toString))

    val labrows = CSVUtils.loadCSVAsTable(sqlContext,"data/lab_results_INPUT.csv","lab")
    val labrowsSQL = sqlContext.sql("SELECT Member_ID, Date_Resulted, Result_Name, Numeric_Result FROM lab WHERE Numeric_Result!=''")
    val labResult: RDD[LabResult] =  labrowsSQL.map(line=>LabResult(line(0).toString, dateFormat.parse(line(1).toString), line(2).toString, line(3).toString.filterNot(",".toSet).toDouble))

    val diag = CSVUtils.loadCSVAsTable(sqlContext,"data/encounter_INPUT.csv","diag")
    val ICD = CSVUtils.loadCSVAsTable(sqlContext,"data/encounter_dx_INPUT.csv","ICD")
    val diagSQL = sqlContext.sql("SELECT Member_ID, Encounter_ID, Encounter_DateTime FROM diag")
    val codeSQL = sqlContext.sql("SELECT Encounter_ID, code FROM ICD")

    val diagICD = sqlContext.sql("SELECT d.Member_ID, d.Encounter_DateTime, i.code FROM diag d JOIN ICD i on d.Encounter_ID = i.Encounter_ID")
    val diagnostic: RDD[Diagnostic] =  diagICD.map(line=> Diagnostic(line(0).toString, dateFormat.parse(line(1).toString), line(2).toString))

    (medication, labResult, diagnostic)
  }

  def createContext(appName: String, masterUrl: String): SparkContext = {
    val conf = new SparkConf().setAppName(appName).setMaster(masterUrl)
    new SparkContext(conf)
  }

  def createContext(appName: String): SparkContext = createContext(appName, "local")

  def createContext: SparkContext = createContext("CSE 8803 Homework Two Application", "local")
}
