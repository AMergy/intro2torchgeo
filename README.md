# intro2torchgeo

# A Github Repository to present some of TorchGeo's applications to the W+ in Geospatial community

The legacy project idea is to understand, set-up and present the library TorchGeo to the Women+ in Geospatial comunity. This understanding would be presented in the form of a github repository, in order to make coding more accessible to the community. One of the main challenges of Geospatial Data Science being the access to computing resources, the library is presented here through a transfer learning project. Transfer learning is also very useful when working with very small labeled dataset, which is often the case in Remote Sensing projects. 

what is a reasonable size and distribution for this dataset - rpz a reasonable amount of variation - 

what would be
knowing what you can change / what you can tune

To change input / outputs, freeze the backbone of the model
- norm max suppression threshold, iou threshold. same structure, but related to the function of the model. 
- for segmentic segmnetation, you can play with loss functions: dice loss (balances overlap between the quantities of the classes) + look at how you do the class weighting. Changing number and type of layers? Backpropagation algorithm?

First choose models and datasets. Then how easy is it to apply: number of channels, resolution, etc

One example of playing with data to improve accuracy (as not much training power), and one example of fine tuning the model as a comparison (but need for computing power)

1. data exploration
2. one approach
3. an improvement of the approach

include chatGPT generated report about how to apply transfer learning on geospatial data

structure: https://drivendata.github.io/cookiecutter-data-science/

# Getting started

An [MLHUB](https://mlhub.earth/) account is required to retrieve Cyclone data from from `torchgeo.datamodules`. To access your API key, create an account and save your key as an environmental vriable, under the name of `MLHUB_API_KEY`. 

# Contact person

- Anne Mergy
- Katie Awty-Carroll

# Ideas

1. **Explore transfer learning as an answer to resource limitations for Geospatial Data Sciences**