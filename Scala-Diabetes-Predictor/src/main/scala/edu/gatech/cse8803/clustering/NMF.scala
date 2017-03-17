package edu.gatech.cse8803.clustering

/**
  * @author Tian Tan <ttan40@gatech.edu>
  */


import breeze.linalg.{sum, DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics.abs
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.Matrices

object NMF {

  /**
   * Run NMF clustering 
   * @param V The original non-negative matrix 
   * @param k The number of clusters to be formed, also the number of cols in W and number of rows in H
   * @param maxIterations The maximum number of iterations to perform
   * @param convergenceTol The maximum change in error at which convergence occurs.
   * @return two matrixes W and H in RowMatrix and DenseMatrix format respectively 
   */
  def run(V: RowMatrix, k: Int, maxIterations: Int, convergenceTol: Double = 1e-4): (RowMatrix, BDM[Double]) = {

    /**
      * TODO 1: Implement your code here
      * Initialize W, H randomly
      * Calculate the initial error (Euclidean distance between V and W * H)
      */

    var W = new RowMatrix(V.rows.map(_ => BDV.rand[Double](k)).map(fromBreeze))
    var H = BDM.rand[Double](k, V.numCols().toInt)
    W.rows.cache()
    V.rows.cache()

    def substraction(V: RowMatrix,WH: RowMatrix): Double = {
        val errorReduce = V.rows.zip(WH.rows).map(line => toBreezeVector(line._1) :- toBreezeVector(line._2)).map(x=> x:*x).map(y=> sum(y)).reduce(_+_)
        errorReduce*0.5
    }

    val initialError = substraction(V,multiply(W,H))
    println("Initial Eorror: "+initialError)
    /**
      * TODO 2: Implement your code here
      * Iteratively update W, H in a parallel fashion until error falls below the tolerance value
      * The updating equations are,
      * H = H.* W^T^V ./ (W^T^W H)
      * W = W.* VH^T^ ./ (W H H^T^)
      */

    var ErrorPre = 0.0
    var iterations = 0
    var ErrorNow = initialError
     while ((ErrorNow-ErrorPre)> convergenceTol & iterations < maxIterations){
       W.rows.cache()
       V.rows.cache()
       ErrorPre = ErrorNow
       val WTV = computeWTV(W,V)
       val WTW = computeWTV(W,W)
       val WTWH = WTW * H
       val mul = WTV :/ (WTWH :+ 2.0e-15)
       H = H :* mul

       val VHT = multiply(V,H.t)
       val HHT = H*H.t
       val WTHS = multiply(W,HHT)
       val mulW = dotDiv(VHT,WTHS)
       W = dotProd(W,mulW)

       val newMatrix = multiply(W,H)
       ErrorNow = substraction(V,newMatrix)
       iterations = iterations + 1

       W.rows.unpersist(false)
       V.rows.unpersist(false)
       println("Current Error: "+ ErrorNow + " Iterations: " +iterations)

     }

    /** TODO: Remove the placeholder for return and replace with correct values */
    (W, H)
  }


  /**  
  * RECOMMENDED: Implement the helper functions if you needed
  * Below are recommended helper functions for matrix manipulation
  * For the implementation of the first three helper functions (with a null return), 
  * you can refer to dotProd and dotDiv whose implementation are provided
  */
  /**
  * Note:You can find some helper functions to convert vectors and matrices
  * from breeze library to mllib library and vice versa in package.scala
  */

  /** compute the mutiplication of a RowMatrix and a dense matrix */
  def multiply(X: RowMatrix, d: BDM[Double]): RowMatrix = {
    val result = X.multiply(fromBreeze_matrix(d))
    result
  }

 /** get the dense matrix representation for a RowMatrix */
  def getDenseMatrix(X: RowMatrix): BDM[Double] = {
    null
  }

  /** matrix multiplication of W.t and V */
  def computeWTV(W: RowMatrix, V: RowMatrix): BDM[Double] = {
    val result = W.rows.zip(V.rows).map{line=>
      val WT = new BDM[Double](line._1.size,1,line._1.toArray)
      val V = new BDM[Double](1,line._2.size,line._2.toArray)
      val prod = WT * V
      (prod)
    }
    result.reduce(_+_)
  }

  /** dot product of two RowMatrixes */
  def dotProd(X: RowMatrix, Y: RowMatrix): RowMatrix = {
    val rows = X.rows.zip(Y.rows).map{case (v1: Vector, v2: Vector) =>
      toBreezeVector(v1) :* toBreezeVector(v2)
    }.map(fromBreeze)
    new RowMatrix(rows)
  }

  /** dot division of two RowMatrixes */
  def dotDiv(X: RowMatrix, Y: RowMatrix): RowMatrix = {
    val rows = X.rows.zip(Y.rows).map{case (v1: Vector, v2: Vector) =>
      toBreezeVector(v1) :/ toBreezeVector(v2).mapValues(_ + 2.0e-15)
    }.map(fromBreeze)
    new RowMatrix(rows)
  }
}
