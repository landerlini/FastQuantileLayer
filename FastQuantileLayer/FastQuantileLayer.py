from __future__ import print_function 
from __future__ import absolute_import 

import tensorflow as tf
import numpy as np 

from scipy.special import erfinv as ierf 
from scipy.linalg  import sqrtm

try: 
  from FixedBinInterpolator import FixedBinInterpolator 
except: 
  from .FixedBinInterpolator import FixedBinInterpolator 


class FastQuantileLayer ( tf.keras.layers.Layer ) :
  """
    Creates a keras layer to emulate the behaviour of 
    scikit-learn QuantileTransformer.
  """

  def __init__ (self, 
      n_quantiles = 50, 
      n_samples   = 200, 
      output_distribution='uniform', 
      default_to_inverse = False, 
      numpy_dtype = np.float32, 
      verbose = False,
      decorrelate = False, 
      **kwargs
    ):
    """
      n_quantiles : int (default: 100)
        Number of quantiles to be computed. It corresponds to 
        the number of landmarks used to discretize the cumulative 
        density function.

      n_sample : int (default: 5000)
        Number of points used to sample the transforms.
        Larger values will result in slower evaluation but more 
        accurate function representation and inversion. 

      output_distribution : string (default: 'uniform')
        Marginal distribution for the transformed data. 
        The choices are 'uniform' (default) or 'normal'.
        The normal distribution is truncated. 

      dtype : numpy data type (default: np.float32)
        Data type of the expected input 

      decorrelate : bool
        If true, after the quantile transform, a linear transform is applied
        to remove the correlation between variables 

      default_to_inverse : bool
        If default_to_inverse is True, and inverse is explicitely specified
        when applying the layer. 
    """

    self._Nbins             = n_quantiles
    self._Nsamples          = n_samples
    self._outDist           = output_distribution
    self.default_to_inverse = default_to_inverse
    self.numpy_dtype        = numpy_dtype 
    self.verbose            = verbose 
    self.decorrelate        = decorrelate

    self.fwdTransforms_ = [] 
    self.bwdTransforms_ = [] 

    self.mean_transformed     = np.array([])
    self.covariance_matrix    = np.array([])
    self.inverse_covmat       = np.array([])       

    tf.keras.layers.Layer.__init__ ( self, kwargs ) 

  def fit ( self, X, y = None ): 
    """
      Creates the tensorflow interpolator used to transform the 
      distribution to either a uniform or normal distribution.  
    """
    rank = len(X.shape) 
    if rank == 1:   # single variable  
      self._fit_column ( X, y ) 

    elif rank == 2: # dataset  
      for iCol in range ( X.shape[1] ): 
        self._fit_column ( X[:,iCol], y ) 
    else:
      raise ValueError ("Expected a numpy array of rank 1 or 2, got %d"%rank)

    if rank == 2 and self.decorrelate:
      t = self.fwdTransforms_
      tX = np.stack([
        np.interp ( X[:,i], np.linspace(t[i].x_min, t[i].x_max, len(t[i].y_values)), t[i].y_values)
        for i in range(X.shape[1]) ]) 

      mean   = np.mean ( tX, axis=1 ) 
      covmat = np.cov ( tX )
      invcov = np.linalg.inv ( covmat ) 

      self.mean_transformed  = mean.astype(self.numpy_dtype)
      self.covariance_matrix = sqrtm(covmat).astype(self.numpy_dtype)
      self.inverse_covmat    = sqrtm(invcov).astype(self.numpy_dtype)

    return self 


  def build ( self, input_shape ):
    tf.keras.layers.Layer.build ( self, input_shape ) 
  

  def _fit_column ( self, X, y=None ):
    """
      Internal. Creates the interpolator for a single variable 
    """

    y  = np.linspace ( 0, 1, self._Nbins ) 
    xq = np.quantile ( X, y )
    if  self._outDist == 'normal' :
      y = ierf ( np.clip(2.*y - 1.,-0.99999, 0.99999)) * np.sqrt(2)

    self.fwdTransforms_ . append (
        FixedBinInterpolator ( xq[0], xq[-1],
          np.interp ( np.linspace(xq[0], xq[-1], self._Nsamples), xq, y ).astype(self.numpy_dtype)
          )
        )

    if self._outDist == 'uniform': 
      self.bwdTransforms_ . append ( 
          FixedBinInterpolator ( y[0], y[-1], xq.astype(self.numpy_dtype) )
        )
    else: 
      self.bwdTransforms_ . append ( 
          FixedBinInterpolator ( y[0], y[-1],
            np.interp ( np.linspace(y[0], y[-1], self._Nsamples), y, xq ).astype(self.numpy_dtype)
            )
          )



  def transform ( self, X, inverse = False, force_decorrelate = None ) : 
    """
      Apply the tensorflow graph 
    """
    if self.default_to_inverse:
      inverse = not inverse

    transf = self.bwdTransforms_ if inverse else self.fwdTransforms_
    rank = len(X.shape) 

    decorrelate = force_decorrelate if force_decorrelate is not None else self.decorrelate
    if rank != 2: self.decorrelate = decorrelate = False

    if not len(transf): 
      raise RuntimeError ( "QuantileTransformTF was not initialized. Run qtf.fit(numpy_dataset)." ) 

    if self.verbose:
      print ("Expected %d columns, got %d." % ( len(transf), X.shape[1]) )

    if inverse and decorrelate:
      X = tf.matmul ( X, self.covariance_matrix ) + self.mean_transformed  

    if rank == 1:
      tX = transf[0].apply ( X[:,i] )  
    elif rank == 2:
      tX = tf.stack ( 
        [ transf[i].apply ( X[:,i] ) for i in range(X.shape[1]) ], 
        axis=1
      )

    if not inverse and decorrelate:
      tX = tf.matmul ( tX - self.mean_transformed , self.inverse_covmat )

    return tX 
      

  def call ( self, X ):
    """
      Service function to call transform 
    """
    return self.transform ( X )

  
  def get_inverse ( self ):
    """
      Return a clone of this layer. 
    """
    new_layer = self.from_config ( self . get_config() ) 
    new_layer . default_to_inverse = not new_layer . default_to_inverse
    return new_layer 

  
  def get_config ( self ):
    """
      Returns the configuration dictionary.
    """
    cfg = tf.keras.layers.Layer.get_config ( self )
    cfg . update ( dict(
        _Nbins             = int(self._Nbins)           ,   
        _Nsamples          = int(self._Nsamples )       , 
        _outDist           = str(self._outDist)         , 
        numpy_dtype        = str(np.dtype(self.numpy_dtype).name) , 
        default_to_inverse = bool(self.default_to_inverse) ,   
        decorrelate        = bool(self.decorrelate), 
        mean_transformed   = self.mean_transformed.tolist(),
        covariance_matrix  = self.covariance_matrix.tolist(),
        inverse_covmat     = self.inverse_covmat.tolist(), 
        direct_transforms  = [
          transform.get_config() for transform in self.fwdTransforms_
        ],
        inverse_transforms = [
          transform.get_config() for transform in self.bwdTransforms_
        ],
    ))
    return cfg 

  
  @classmethod
  def from_config ( cls, cfg ):
    """
      Returns the configuration dictionary.
    """
    newLayer = FastQuantileLayer() 
    newLayer._Nbins               = cfg [ '_Nbins' ] 
    newLayer._Nsamples            = cfg [ '_Nsamples' ] 
    newLayer.numpy_dtype          = cfg [ 'numpy_dtype'] 
    newLayer.default_to_inverse   = cfg [ 'default_to_inverse' ] 
    newLayer.decorrelate          = bool(cfg [ 'decorrelate' ])
    newLayer.mean_transformed     = np.array(cfg [ 'mean_transformed' ]).astype(newLayer.numpy_dtype) 
    newLayer.covariance_matrix    = np.array(cfg [ 'covariance_matrix' ]).astype(newLayer.numpy_dtype)
    newLayer.inverse_covmat       = np.array(cfg [ 'inverse_covmat' ]).astype(newLayer.numpy_dtype)
    newLayer.fwdTransforms_       = [] 
    newLayer.bwdTransforms_       = [] 
    
    for transform in cfg [ 'direct_transforms' ]:
      newLayer.fwdTransforms_ . append ( 
        FixedBinInterpolator ( transform['x_min'], transform['x_max'], 
          np.array(transform['y_values'], dtype=transform ['dtype'] ))
      )

    for transform in cfg [ 'inverse_transforms' ]:
      newLayer.bwdTransforms_ . append ( 
        FixedBinInterpolator ( transform['x_min'], transform['x_max'], 
          np.array(transform['y_values'], dtype=transform ['dtype'] ))
      )

    return newLayer


  def compute_output_shape ( self, input_shape ):
    return input_shape



if __name__ == '__main__':
  dataset = np.c_[
    np.random.uniform ( 0., 1., 1000) , 
    np.random.uniform ( -5., 50., 1000) , 
  ]
  th = np.pi / 5. 
  rotmat = np.array([[np.cos(th),np.sin(th)],[-np.sin(th),np.cos(th)]])
  dataset = np.matmul ( dataset, rotmat ) 

  transformer = FastQuantileLayer (output_distribution='normal', decorrelate=False)
  transformer . fit ( dataset ) 

  transformer . from_config ( transformer.get_config() ) 

  test_dataset = tf.constant(
    np.matmul ( np.c_[
      np.random.uniform ( 0., 1., 10000) , 
      np.random.uniform ( -5, 50., 10000) , 
    ],rotmat), dtype = tf.float32)

  t = transformer . transform ( test_dataset ) 

  bkwd = transformer . get_inverse() . transform ( t )

  with tf.Session() as session: 
    print ("###### Original dataset ####### " ) 
    print ("Mean: ", np.mean ( test_dataset.eval() , axis= 0) ) 
    print ("Std: ",  np.std  ( test_dataset.eval() , axis= 0) ) 
    print () 
    print ("###### Forward transform ####### " ) 
    print ("Mean:", np.mean((t.eval()), axis= 0))
    print ("Std: ", np.std ((t.eval()), axis= 0))
    print ("CovMat: ", np.cov ( t.eval(), rowvar = False ) )
    print () 
    print ("###### Backward transform ####### " ) 
    print ("Mean: ", np.mean ( bkwd.eval() , axis= 0) ) 
    print ("Std: ",  np.std  ( bkwd.eval() , axis= 0) ) 
    print () 
    print ("Average squared error: ", np.sqrt(np.mean ( np.square ( test_dataset.eval() - bkwd.eval() ) ))) 
    print ("Max.    squared error: ", np.sqrt(np.max  ( np.square ( test_dataset.eval() - bkwd.eval() ) ))) 
    cmpr = np.c_[test_dataset.eval(), t.eval(),  bkwd.eval()] 
    error = np.abs(cmpr[:,0]-cmpr[:,4]) 

    print ( "Largest errors: " ) 
    print (cmpr [np.argsort(-error)][:10] ) 
    

    
  
  
