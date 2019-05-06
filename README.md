# FastQuantileLayer
FastQuantileLayer is a Layer for Keras implementing the QuantileTransform 
similarly to scikit-learn QuantileTransformer. 
A similar implementation, more precise but not bound to Keras, can be found here: 
  https://github.com/yandexdataschool/QuantileTransformerTF/blob/master/README.md

The purpose of this package is:
 - remove all dependencies on scikit-learn
 - obtain an evaluation of the direct and inverse transform as fast as possible 
   (trading some precision for performance)
 - obtain a TensorFlow graph runnable in a Sequential model in Keras 


The package is composed of two classes:
 - *FixedBinInterpolator*: intended to interpolate a point-defined function
   y = f(x) with equidistant x samples (x-grid)
 - *FastQuantileLayer*: intended to compute the transform to preprocess
   the input data into a uniform- or normal-distributed variable. 


### Example outside Keras 

```
  ## Creates the training dataset 
  dataset = np.random.uniform ( 0., 1., 1000 ) 

  ## Train the QuantileTransformer 
  transformer = FastQuantileLayer (output_distribution='normal')
  transformer . fit ( dataset ) 

  ## Gets a new dataset with the same distribution as the training dataset
  test_dataset = tf.constant(np.random.uniform ( 0., 1., 100000 ))

  ## Transform the variable into a Gaussian-distributed variable t 
  t = transformer . transform ( test_dataset ) 
  
  [...] 

  ## Appiles the inverted transform to the Gaussian distributed variable t  
  bkwd = transformer . transform ( t, inverse=True ) 

  ## bkwd differs from test_dataset only for computational errors 
  ## (order 1e-5) that can be reduced tuning the arguments of QuantileTransformer
```


### Example within Keras 

```
  ## Creates the training dataset 
  dataset = np.random.uniform ( 0., 1., 1000 ) 

  model = tf.keras.models.Sequential()
  model.add ( FastQuantileLayer ( output_distribution = 'normal' ).fit ( dataset ) )
  model.add ( Dense ( 10, activation = 'tanh' ) )
  model.add ( Dense ( 1, activation = 'sigmoid' ) ) 

  
```
