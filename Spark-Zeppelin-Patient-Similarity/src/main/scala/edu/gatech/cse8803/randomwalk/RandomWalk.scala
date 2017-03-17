package edu.gatech.cse8803.randomwalk

import edu.gatech.cse8803.model.{PatientProperty, EdgeProperty, VertexProperty}
import org.apache.spark.graphx._

object RandomWalk {

  def randomWalkOneVsAll(graph: Graph[VertexProperty, EdgeProperty], patientID: Long, numIter: Int = 100, alpha: Double = 0.15): List[Long] = {
    /** 
    Given a patient ID, compute the random walk probability w.r.t. to all other patients. 
    Return a List of patient IDs ordered by the highest to the lowest similarity.
    For ties, random order is okay
    */

    /** Remove this placeholder and implement your code */
//    List(1,2,3,4,5)
    val patient = graph.vertices.filter(_.2.isInstanceOf[PatientProperty])
    val patientNum = patient.keys.max()
    val src = patientID
    var rankGraph: Graph[Double,Double] = graph
                                          .outerJoinVertices(graph.outDegrees){(vid, vdata, deg)=> deg.getOrElse(0)}
                                          .mapTriplets(e=>1.0/e.srcAttr,TripletFields.Src)
                                          .mapVertices((id,attr)=> if (!(id!=src)) alpha else 0.0)
    def delta(u:VertexId, v:VertexId): Double= {if (u==v) 1.0 else 0.0}
    var iteration = 0
    var prevRankGraph: Graph[Double,Double] = null
    while(iteration<numIter){
      rankGraph.cache()
      val rankUpdates = rankGraph.aggregateMessages[Double](
        ctx=>ctx.sendToDst(ctx.srcAttr*ctx.attr),_+_,TripletFields.Src)
      prevRankGraph = rankGraph
      val rPrb = (src:VertexId, id:VertexId) => alpha*delta(src,id)
      rankGraph = rankGraph.joinVertices(rankUpdates){
        (id,oldRank,msgSum) => rPrb(src,id) + (1.0-alpha) * msgSum
      }.cache()
      rankGraph.edges.foreachPartition(x=>{})
      prevRankGraph.vertices.unpersist(false)
      prevRankGraph.edges.unpersist(false)
      iteration +=1
    }
    val result = rankGraph.vertices.filter(_._1<=patientNum).filter(_._1!=patientID).sortBy(_._2,false).map(x=>x._1).take(10).toList
    result
  }
}
