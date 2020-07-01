/**
 * Copyright 2018 LinkedIn Corporation. All rights reserved. Licensed under the BSD-2 Clause license.
 * See LICENSE in the project root for license information.
 */
package com.linkedin.nn.algorithm

import java.util.Random

import com.linkedin.nn.linalg.RandomProjection
import com.linkedin.nn.lsh.SignRandomProjectionHashFunction
import com.linkedin.nn.model.{DotProductRandomProjectionModel, LSHNearestNeighborSearchModel}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.util.Identifiable

/**
 * This is the gateway that should be used for creating a [[DotProductRandomProjectionModel]] that can be used for
 * performing approximate nearest neighbor search in dot product distance space. The model parameters can be set using the
 * provided setters and then calling [[createModel()]] providing it the input dimension. This creates the hash
 * functions which in this case are sign-[[RandomProjection]] where each element of the projection matrix is sampled
 * from a gaussian distribution.
 */
class DotProductRandomProjectionNNS(override val uid: String = Identifiable.randomUID("DotRandomProjectionLSH"))
  extends LSHNearestNeighborSearch[DotProductRandomProjectionModel] {
  override def setNumHashes(n: Int): this.type = super.setNumHashes(n)

  override def setJoinParallelism(parallelism: Int): this.type = super.setJoinParallelism(parallelism)

  override def setBucketLimit(limit: Int): this.type = super.setBucketLimit(limit)

  override def setSignatureLength(r: Int): this.type = super.setSignatureLength(r)

  /**
   * Create a [[DotProductRandomProjectionNNS]]
   * @param dimension The dimension of vectors in the input dataset
   * @return A new [[DotProductRandomProjectionNNS]]
   */
  override def createModel(dimension: Int): LSHNearestNeighborSearchModel[DotProductRandomProjectionModel] = {
    val rand = new Random($(seed))
    val randProjections: Array[SignRandomProjectionHashFunction] = {
      Array.fill($(numHashes) / $(signatureLength)) {
        new SignRandomProjectionHashFunction(RandomProjection.generateGaussian(dimension, $(signatureLength), rand))
      }
    }
    val model = new DotProductRandomProjectionModel(uid, randProjections)
    copyValues(model)
    model
  }

  override def copy(extra: ParamMap): Params = defaultCopy(extra)
}
