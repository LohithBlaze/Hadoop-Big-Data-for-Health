/**

students: please put your implementation in this file!
  **/
package edu.gatech.cse8803.jaccard

import edu.gatech.cse8803.model._
import edu.gatech.cse8803.model.{EdgeProperty, VertexProperty}
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD


object Jaccard {

  def jaccardSimilarityOneVsAll(graph: Graph[VertexProperty, EdgeProperty], patientID: Long): List[Long] = {
    /** 
    Given a patient ID, compute the Jaccard similarity w.r.t. to all other patients. 
    Return a List of top 10 patient IDs ordered by the highest to the lowest similarity.
    For ties, random order is okay. The given patientID should be excluded from the result.
    */

    /** Remove this placeholder and implement your code */
//    List(1,2,3,4,5)

    val NeighborIDs = graph.collectNeighborIds(EdgeDirection.Either).map(x=>(x._1,x._2.filter(y=>y>1000))).filter(_._1<=1000)
    val Noself = NeighborIDs.filter(_._1 != patientID)
    val Self = graph.collectNeighborIds(EdgeDirection.Either).filter(_._1 == patientID).map(x=>x._2).collect.flatten.toSet
    val simu = Noself.map(line=>(line._1,jaccard(Self,line._2.toSet)))
    val result= simu.sortBy(_._2,false).map(x=>x._1).take(10).toList
    result
  }

  def jaccardSimilarityAllPatients(graph: Graph[VertexProperty, EdgeProperty]): RDD[(Long, Long, Double)] = {
    /**
    Given a patient, med, diag, lab graph, calculate pairwise similarity between all
    patients. Return a RDD of (patient-1-id, patient-2-id, similarity) where 
    patient-1-id < patient-2-id to avoid duplications
    */

    /** Remove this placeholder and implement your code */
//    val sc = graph.edges.sparkContext
//    sc.parallelize(Seq((1L, 2L, 0.5d), (1L, 3L, 0.4d)))
      val NeighborIDs = graph.collectNeighborIds(EdgeDirection.Either).map(x=>(x._1,x._2.filter(y=>y>1000))).filter(_._1<=1000)
      val Allpairs = NeighborIDs.cartesian(NeighborIDs).filter{case(a,b)=> a._1<b._1}
      val result = Allpairs.map(x=>(x._1._1,x._2._1,jaccard(x._1._2.toSet,x._2._2.toSet)))
      result
  }

  def jaccard[A](a: Set[A], b: Set[A]): Double = {
    /** 
    Helper function

    Given two sets, compute its Jaccard similarity and return its result.
    If the union part is zero, then return 0.
    */

    /** Remove this placeholder and implement your code */
    val unioned:Double = a.union(b).size.toDouble
    val intersected:Double = a.intersect(b).size.toDouble
    return (if(unioned==0) 0.0 else intersected/unioned)
  }

}
