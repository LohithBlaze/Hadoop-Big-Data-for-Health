/**
 * @author Tian Tan
 */
package edu.gatech.cse8803.features

import edu.gatech.cse8803.model.{Diagnostic, LabResult, Medication}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD


object FeatureConstruction {

  /**
   * ((patient-id, feature-name), feature-value)
   */
  type FeatureTuple = ((String, String), Double)

  /**
   * Aggregate feature tuples from diagnostic with COUNT aggregation,
   * @param diagnostic RDD of diagnostic
   * @return RDD of feature tuples
   */
  def constructDiagnosticFeatureTuple(diagnostic: RDD[Diagnostic]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    /* diagnostic.map(line => ((line.patientID,line.code),1.0)).keyBy(f =>f._1).reduceByKey((x,y)=>(x._1,x._2+y._2)).map(v =>v._2)*/
    diagnostic.map(line=>((line.patientID,line.code),1.0)).reduceByKey((x,y)=>x+y)
  }

  /**
   * Aggregate feature tuples from medication with COUNT aggregation,
   * @param medication RDD of medication
   * @return RDD of feature tuples
   */
  def constructMedicationFeatureTuple(medication: RDD[Medication]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    medication.map(line=>((line.patientID,line.medicine),1.0)).reduceByKey((x,y)=>x+y)
  }

  /**
   * Aggregate feature tuples from lab result, using AVERAGE aggregation
   * @param labResult RDD of lab result
   * @return RDD of feature tuples
   */
  def constructLabFeatureTuple(labResult: RDD[LabResult]): RDD[FeatureTuple] = {
    /**
      * TODO implement your own code here and remove existing
      * placeholder code
      */
    labResult.map(line => ((line.patientID, line.testName), (line.value, 1))).reduceByKey((x, y) => ((x._1 + y._1), (x._2 + y._2))).map(v => (v._1, v._2._1 / v._2._2))
  }
  /**
   * Aggregate feature tuple from diagnostics with COUNT aggregation, but use code that is
   * available in the given set only and drop all others.
   * @param diagnostic RDD of diagnostics
   * @param candiateCode set of candidate code, filter diagnostics based on this set
   * @return RDD of feature tuples
   */
  def constructDiagnosticFeatureTuple(diagnostic: RDD[Diagnostic], candiateCode: Set[String]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    diagnostic.filter(line=>candiateCode.contains(line.code)).map(f=>((f.patientID,f.code),1.0)).reduceByKey((x,y)=>x+y)
  }

  /**
   * Aggregate feature tuples from medication with COUNT aggregation, use medications from
   * given set only and drop all others.
   * @param medication RDD of diagnostics
   * @param candidateMedication set of candidate medication
   * @return RDD of feature tuples
   */
  def constructMedicationFeatureTuple(medication: RDD[Medication], candidateMedication: Set[String]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    medication.filter(line=>candidateMedication.contains(line.medicine)).map(f=>((f.patientID,f.medicine),1.0)).reduceByKey((x,y)=>x+y)
  }


  /**
   * Aggregate feature tuples from lab result with AVERAGE aggregation, use lab from
   * given set of lab test names only and drop all others.
   * @param labResult RDD of lab result
   * @param candidateLab set of candidate lab test name
   * @return RDD of feature tuples
   */
  def constructLabFeatureTuple(labResult: RDD[LabResult], candidateLab: Set[String]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    labResult.filter(line=>candidateLab.contains(line.testName)).map(line => ((line.patientID, line.testName), (line.value, 1))).reduceByKey((x, y) => ((x._1 + y._1), (x._2 + y._2))).map(v => (v._1, v._2._1 / v._2._2))
  }


  /**
   * Given a feature tuples RDD, construct features in vector
   * format for each patient. feature name should be mapped
   * to some index and convert to sparse feature format.
   * @param sc SparkContext to run
   * @param feature RDD of input feature tuples
   * @return
   */
  def construct(sc: SparkContext, feature: RDD[FeatureTuple]): RDD[(String, Vector)] = {

    /** save for later usage */
    feature.cache()

    /** create a feature name to id map*/

    /** transform input feature */

    /**
     * Functions maybe helpful:
     *    collect
     *    groupByKey
     */

    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    val featureName = feature.map(line=>line._1._2).distinct().zipWithIndex().collectAsMap()
    val featureNum = featureName.values.size
    val result = feature.map(line=>(line._1._1,featureName(line._1._2),line._2)).groupBy(_._1).map{ x=>
      val featureValue = x._2.map(y=>(y._2.toInt,y._3.toDouble)).toArray
      /* Vectors.sparse(int size, int[] indices, double[] values) create a sparse vector providing its index array and value array*/
      (x._1, Vectors.sparse(featureNum, featureValue))
    }
    /*result.sortByKey().take(10).foreach(println)*/
    result
    /** The feature vectors returned can be sparse or dense. It is advisable to use sparse */

  }
}


