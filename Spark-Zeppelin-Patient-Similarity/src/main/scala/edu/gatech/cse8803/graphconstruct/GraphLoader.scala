/**
 * @author Hang Su <hangsu@gatech.edu>.
 */

package edu.gatech.cse8803.graphconstruct

import edu.gatech.cse8803.model._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD


object GraphLoader {
  /** Generate Bipartite Graph using RDDs
    *
    * @input: RDDs for Patient, LabResult, Medication, and Diagnostic
    * @return: Constructed Graph
    *
    * */


  def load(patients: RDD[PatientProperty], labResults: RDD[LabResult],
           medications: RDD[Medication], diagnostics: RDD[Diagnostic]): Graph[VertexProperty, EdgeProperty] = {

    /** HINT: See Example of Making Patient Vertices Below */
    val vertexPatient: RDD[(VertexId, VertexProperty)] = patients
      .map(patient => (patient.patientID.toLong, patient.asInstanceOf[VertexProperty]))

    val labResultsEarly = labResults.map(line=>((line.patientID,line.labName),line)).reduceByKey((x,y)=> if (x.date>y.date) x else y).map(line=>line._2)
    val diagnosticsEarly =diagnostics.map(line=>((line.patientID,line.icd9code),line)).reduceByKey((x,y)=> if (x.date>y.date) x else y).map(line=>line._2)
    val medicationsEarly = medications.map(line=>((line.patientID,line.medicine),line)).reduceByKey((x,y)=> if (x.date>y.date) x else y).map(line=>line._2)

    val startLab = patients.map(line=>line.patientID).max().toLong+1000

    val labVertexIdRDD = labResultsEarly.map(_.labName).distinct.zipWithIndex.map{case(testName,zeroBasedIndex)=>(testName,zeroBasedIndex+startLab)}
    val labVectexId = labVertexIdRDD.collect.toMap
    val vertexLabResult: RDD[(VertexId, VertexProperty)] = labVertexIdRDD
      .map{case(testName,index)=>(index, LabResultProperty(testName))}.asInstanceOf[RDD[(VertexId,VertexProperty)]]

    val startDiag=startLab+labVertexIdRDD.count()+1000
    val diagVertexIdRDD = diagnosticsEarly.map(_.icd9code).distinct.zipWithIndex.map{case(icd9code,zeroBasedIndex)=>(icd9code,zeroBasedIndex+startDiag)}
    val diagVertexId = diagVertexIdRDD.collect.toMap
    val vertexDiagnostics: RDD[(VertexId, VertexProperty)] = diagVertexIdRDD
      .map{case(icd9code,index)=>(index,DiagnosticProperty(icd9code))}.asInstanceOf[RDD[(VertexId,VertexProperty)]]

    val startMed = startDiag + diagVertexIdRDD.count()+1000
    val medVertexIdRDD = medicationsEarly.map(_.medicine).distinct.zipWithIndex.map{case(medicine,zeroBasedIndex)=>(medicine,zeroBasedIndex+startMed)}
    val medVertexId = medVertexIdRDD.collect.toMap
    val vertexMedications: RDD[(VertexId, VertexProperty)] = medVertexIdRDD
      .map{case(medicine,index)=>(index, MedicationProperty(medicine))}.asInstanceOf[RDD[(VertexId,VertexProperty)]]
    /** HINT: See Example of Making PatientPatient Edges Below
      *
      * This is just sample edges to give you an example.
      * You can remove this PatientPatient edges and make edges you really need
      * */
//    case class PatientPatientEdgeProperty(someProperty: SampleEdgeProperty) extends EdgeProperty
//
//    val edgePatientPatient: RDD[Edge[EdgeProperty]] = patients
//      .map({p =>
//        Edge(p.patientID.toLong, p.patientID.toLong, SampleEdgeProperty("sample").asInstanceOf[EdgeProperty])
//      })

    val sc = patients.sparkContext
    val bclabVertexId = sc.broadcast(labVectexId)
    val bcdiagVertexId = sc.broadcast(diagVertexId)
    val bcmedVertexId = sc.broadcast(medVertexId)

    val edgePatientLab = labResultsEarly.map(line=>(line.patientID,line.labName,line)).map{case(patientID,labName,labresult)=>Edge(patientID.toLong,bclabVertexId.value(labName),PatientLabEdgeProperty(labresult).asInstanceOf[EdgeProperty])}
    val edgeLabPatient = labResultsEarly.map(line=>(line.patientID,line.labName,line)).map{case(patientID,labName,labresult)=>Edge(bclabVertexId.value(labName),patientID.toLong,PatientLabEdgeProperty(labresult).asInstanceOf[EdgeProperty])}

    val edgePatientDiag = diagnosticsEarly.map(line=>(line.patientID,line.icd9code,line)).map{case(patientID,icd9code,diag)=>Edge(patientID.toLong,bcdiagVertexId.value(icd9code),PatientDiagnosticEdgeProperty(diag).asInstanceOf[EdgeProperty])}
    val edgeDiagPatient = diagnosticsEarly.map(line=>(line.patientID,line.icd9code,line)).map{case(patientID,icd9code,diag)=>Edge(bcdiagVertexId.value(icd9code),patientID.toLong,PatientDiagnosticEdgeProperty(diag).asInstanceOf[EdgeProperty])}

    val edgePatientMed = medicationsEarly.map(line=>(line.patientID,line.medicine,line)).map{case(patientID,medicine,med)=>Edge(patientID.toLong,bcmedVertexId.value(medicine),PatientMedicationEdgeProperty(med).asInstanceOf[EdgeProperty])}
    val edgeMedPatient = medicationsEarly.map(line=>(line.patientID,line.medicine,line)).map{case(patientID,medicine,med)=>Edge(bcmedVertexId.value(medicine),patientID.toLong,PatientMedicationEdgeProperty(med).asInstanceOf[EdgeProperty])}

    // Making Graph
    val vertexall = sc.union(vertexPatient,vertexLabResult,vertexDiagnostics,vertexMedications)
    val edgeall = sc.union(edgePatientLab,edgeLabPatient,edgePatientDiag,edgeDiagPatient,edgePatientMed,edgeMedPatient)
    val graph: Graph[VertexProperty, EdgeProperty] = Graph(vertexall, edgeall)
//    graph.vertices.foreach(println)
//    println(graph.numVertices)

    graph
  }
}
