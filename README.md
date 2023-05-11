# Women+ in Geospatial - Mentorship Program

This repository is the legacy project of the cohort #3 of the Mentorship Program W+iG (2022-2023).

During our Mentorship journey, we realized the extent of the problem of having limited resource usage when training a model on Geospatial data. This can be a barrier when it comes to exploring the possibilities offered by data science and understanding different approaches. There are several ways to overcome this issue, and we wish to explore them collectively, in order to allow every WiG+ member wishing to familiarize themselves with deep learning to participate and use current tools to explore their ideas. In the scope of this work, we implemented a transfer learning approach that can be used as a baseline for comparison.

The aim of this repository is to create a community project allowing all members from the WiG+ community (and beyond) to explore an approach and improve the accuracy of the baseline model. We hope to allow as many WiG+ as possible to familiarize themselves with the library [torchgeo](https://torchgeo.readthedocs.io/en/stable/) and with good practices when it comes to contributing to a github project.

# Study

This code was implemented and run on a system with the following characteristics:
- number of CPUs
```
CPU(s):                  8
```
- RAM
```
Mem:            15Gi
Swap:          2,0Gi
```

A choice was made to not use any GPU, to make sure this code can be run by any user on most systems, and to make sure satisfying results can be reached with limited resources. 

## Dataset

- The [NAIP (National Agriculture Imagery Program)](https://catalog.data.gov/dataset/national-agriculture-imagery-program-naip) dataset is a high-resolution aerial imagery dataset of the United States, with four spectral bands (R, G, B, IR). It can be used for various applications including land use classification, crop mapping, and object detection.

- The [Chesapeake](https://www.chesapeakeconservancy.org/wp-content/uploads/2017/01/LandCover101Guide.pdf) dataset is a comprehensive land cover dataset covering the Chesapeake Bay Watershed region of the United States. The dataset includes a wide range of land cover classes, such as forests, wetlands, croplands, and urban areas, among others. It is a valuable resource for a variety of applications, including land use planning, conservation, and natural resource management.

We are using NAIP data's images and Chesepeake's labels. 

## Model

ResNet-50 is a deep neural network architecture that is widely used for image classification and computer vision tasks. It was introduced by Microsoft Research in 2015 and has since become one of the most popular deep learning models in the field.

ResNet-50 is based on a deep residual learning framework, which allows the model to be trained much deeper than traditional neural networks without encountering the problem of vanishing gradients. The model consists of 50 layers and uses skip connections to allow information to flow directly from one layer to another, thereby enabling the model to learn more efficiently.

The architecture of ResNet-50 includes convolutional layers, batch normalization layers, max pooling layers, and fully connected layers. The convolutional layers extract features from the input image, while the fully connected layers provide the final output probabilities for each class. The batch normalization layers help to stabilize the training process and improve the accuracy of the model.

ResNet-50 has achieved state-of-the-art performance on a variety of image classification benchmarks, including the ImageNet dataset, which contains over a million images across 1,000 categories. The model has also been used for a range of other computer vision tasks, such as object detection, segmentation, and image captioning.

## Objective

This legacy project has 3 main objectives:
- get familiarized with Torchgeo
- give the opportunity to the WiG+ to get started with Geospatial Data Sciences
- explore and identify the best ways to tackle resource limits when training a Deep Learning algorithm, as a comunity

Potential applications could include different situations. For instance:
1. Accessible Deep Learning: This project enables individuals and organizations without access to a powerful GPU or computing resources to train deep learning algorithms on geospatial data. This can help democratize the field of data science and make it more accessible to a wider range of people and organizations, including those with limited financial resources.

2. Embedded Systems: This project can be used to train deep learning models for use in embedded systems, such as onboard satellites or in edge devices, where computational resources are limited. The use of transfer learning and other techniques to optimize model performance with limited resources can be particularly valuable in these contexts, helping to reduce costs and improve performance.

3. Environmental Monitoring: The project can be applied to train deep learning models to analyze geospatial data for environmental monitoring purposes, such as monitoring vegetation cover, water levels, and air quality. By using transfer learning and other techniques to optimize model performance with limited resources, it becomes possible to perform these tasks on a personal laptop or low-cost computing devices.

4. Disaster Response: The project can be used to train deep learning models to analyze satellite imagery and support disaster response efforts. This includes identifying areas of damage and assessing the impact of natural disasters, helping emergency responders target their efforts more effectively, even in situations where traditional computing resources are unavailable.

5. Agricultural Applications: The project can be used to train deep learning models to analyze geospatial data for agricultural purposes, such as monitoring crop growth and soil moisture levels. By using transfer learning and other techniques to optimize model performance with limited resources, it becomes possible to perform these tasks on a personal laptop or low-cost computing devices, helping to reduce costs and improve agricultural productivity.

One of the expected impacts of this project is to make coding and deep learning more accessible to the comunity, by presenting it in a clear, simple and concise way. 


# Structure

# :rocket: Getting started

1. Clone the repository and change directory: `cd intro2torchgeo`
2. Create a new conda environment: `conda env create -f environment.yml`
3. Run the notebook `1_data_exploration.ipynb` to understand the datasets' structure and torchgeo's custom classes
4. Run the notebook `2_classification_baseilne.ipynb` to train a classification model with transfer learning and get a first version of model weights which can be used as a template


# :pencil: Contributing

You are now ready for contributing!

Think about how you could improve the performance of this model without having much computing power (for instance by using vegetation indices, by playing with the loss function, the number and type of layers).

Once you have an idea, make sure you follow these steps:
1. Create a detailed issue and a new branch
2. Switch to the new branch locally on your machine, and start a new notebook with your idea. Make sure you document your code and decisions.
3. Train your model, and evaluate it using the evaluation section. Make sure you add your idea and the new performance in the table below.
4. Make the PR - make sure you detail your decisions.

All **suggestions/PRs** are welcome!

# Results summary

| Name | Test Loss | Method description |
|--------|--------|--------|
| AMergy | 1595.2490 | Transfer learning |
| ... | ... | ... |
| ... | ... | ... |

If needed, detail the method you investigated in the section below:
- **Transfer learning**: modified version of the FarSeg segmentation model with a ResNet50 backbone that is pre-trained on a large dataset. The first layer of the model is modified to accept a 4-band image.
- **...**: ...

# Contacts

- [Anne Mergy](https://github.com/AMergy)
- [Katie Awty-Carroll](https://github.com/klh5)
