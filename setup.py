import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='FastQuantileLayer',  
     version='0.1',
     scripts=[] ,
     author="Lucio Anderlini",
     author_email="l.anderlini@gmail.com",
     description="Keras Layer to apply Quantile transform and its inverse",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/landerli/FastQuantileLayer",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )

