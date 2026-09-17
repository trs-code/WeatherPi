# WeatherPi

A C-based neural-network library and embedded weather inference system built around a **BME280 environmental sensor**, custom neural-network primitives, recurrent context windows, serialized CML models, and generated C model representations.

The project combines a low-level neural-network implementation with a concrete weather-classification application intended for deployment on Linux/embedded hardware.

**Copyright © 2025 Tafadar Soujad**

Licensed under the **GNU General Public License v3.0**. See [`LICENSE.txt`](LICENSE.txt).

---

## Overview

WeatherPi consists of two closely related components:

1. A general-purpose C neural-network implementation for constructing, training, serializing, loading, and executing neural-network models.
2. A weather-sensing application that reads **temperature, pressure, and humidity** from a BME280 sensor over Linux I²C and feeds those measurements into a trained recurrent neural network.

The primary application entry point is [`operate.c`](operate.c).

The deployed execution path is:

```text
BME280
   │
   │ I²C
   ▼
sensor.h
   │
   │ temperature / pressure / humidity
   ▼
operate.c
   │
   │ 3-element input vector
   ▼
RNN Weather Model
   │
   │ inference
   ▼
2-element output
```

The repository also contains serialized CML models and generated C representations of trained models for deployment.

---

# Repository Architecture

```text
WeatherPi/
│
├── dataset/
│   └── Weather datasets and preprocessing scripts
│
├── src/
│   ├── include/
│   │   ├── activation.h
│   │   ├── cml.h
│   │   ├── helper_funcs.h
│   │   ├── layer.h
│   │   ├── layer_construct.h
│   │   ├── layer_destruct.h
│   │   ├── loss.h
│   │   ├── model.h
│   │   ├── model_construct.h
│   │   ├── model_destruct.h
│   │   ├── model_ops.h
│   │   ├── nn_math.h
│   │   ├── nn_ops.h
│   │   └── sensor/
│   │       └── sensor.h
│   │
│   └── tests/
│       └── Test data and model fixtures
│
├── operate.c
│
├── weatherPiModel.c
├── weatherPiModelContext.c
│
├── weathrModel.cml
├── weathrModelContext.cml
├── weathrModelContextBest.cml
├── weathrPiModelBest.cml
│
├── Repo Diagram.png
├── BME280 Sensor Manual.pdf
└── LICENSE.txt
```

The repository currently separates the neural-network implementation into model management, layer management, mathematical operations, activations, losses, serialization, helper functionality, and sensor access.

---

# Neural-Network Architecture

## Layer Representation

The fundamental unit of the neural-network implementation is the `layer` structure.

Each layer stores its:

* Number of nodes
* Number of predecessor nodes
* Number of predecessor layers
* Weights
* Biases
* Outputs
* Pre-activations
* Backpropagation errors
* Activation derivatives
* Layer ID
* Activation-function identifier
* Layer-type identifier
* References to predecessor layers

The current implementation uses `float` for neural-network numerical data.

A simplified representation is:

```text
layer
├── weights
├── biases
├── outputs
├── preActivations
├── backErrors
├── activationDerivatives
├── prevLayers
├── numNodes
├── numPrevNodes
├── numPrevLayers
├── layerID
├── activationFunction
└── layerType
```

### Pointer-Based Ownership

The library intentionally operates on **pointers to allocated layer pointers**.

This results in interfaces using `layer***` in places where the library needs to manipulate the caller's layer allocation.

The design allows the library to:

* Operate on caller-owned layer instances
* Destroy layers through library functions
* Invalidate the caller's original pointer
* Reduce the possibility of double frees
* Reduce the possibility of dangling layer pointers

This is an intentional low-level memory-management decision rather than an attempt to hide allocation from the user.

---

# Graph-Based Models

Layers are not restricted to a sequential chain.

A layer may have multiple predecessor layers, allowing models to represent branching and merging computational structures.

Conceptually:

```text
        ┌──────────┐
        │ Input A  │
        └────┬─────┘
             │
             ▼
        ┌──────────┐
        │ Hidden A │───────┐
        └──────────┘       │
                           ▼
                      ┌─────────┐
        ┌──────────┐  │ Output  │
        │ Hidden B │─►│         │
        └────┬─────┘  └─────────┘
             ▲
             │
        ┌────┴─────┐
        │ Input B  │
        └──────────┘
```

This architecture provides a foundation for networks containing more complicated connectivity than a traditional sequential stack.

---

# Topological Execution

The model originally relied on recursive graph traversal to determine the order in which layers were evaluated.

Forward propagation required predecessor layers to be evaluated before their successors, while backpropagation required the reverse ordering.

The current implementation replaces repeated recursive traversal with an explicitly maintained layer list.

The `model` structure contains:

```text
numLayers
numInLayers
layerList
inLayers
outLayer
lossDerivatives
targets
learningRate
loss_fn
```

In particular:

* `layerList` stores the model's layers in execution order.
* `inLayers` provides references to the model's input layers.
* `outLayer` references the final output layer.

The layer list is topologically ordered so that predecessor layers occur before successor layers.

This changes model execution from repeated recursive traversal into an iterative pass through an already-established execution order.

```text
Graph construction
       │
       ▼
Topological sorting
       │
       ▼
Ordered layer list
       │
       ▼
Iterative model execution
```

This was done to reduce recursive traversal overhead and make the implementation more compatible with a data-oriented execution model.

The resulting approach has several advantages:

* Explicit execution order
* Iterative traversal
* Reduced stack usage
* Better spatial locality
* Separation of graph construction from execution
* A simpler foundation for low-level optimization

The repository's `Repo Diagram.png` provides a visual representation of the project architecture.

---

# Activation Functions

Activation functions and their derivatives are implemented in [`activation.h`](src/include/activation.h).

The current implementation includes:

### Standard activations

* Linear
* ReLU
* Leaky ReLU
* Tanh
* Sigmoid
* Softmax

### Fast approximations

* Fast tanh
* Fast sigmoid
* Fast softmax

Fast approximations are implemented using custom numerical approximations rather than directly relying on the corresponding standard-library functions.

Activation functions are selected through a compact character-based identifier stored by each layer.

---

# Loss Functions

Loss functions are implemented in [`loss.h`](src/include/loss.h).

The current implementation provides:

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* Mean Bias Error (MBE)
* Huber loss
* Binary Cross Entropy
* Categorical Cross Entropy

Fast implementations are also provided for binary and categorical cross entropy.

Each supported loss has an associated derivative used during model training and backpropagation.

---

# Recurrent Neural Networks

The neural-network architecture was extended to support recurrent behavior through **context windows**.

A context window creates additional time-step representations that retain information from previous inputs while sharing weights with the layer whose context is being extended.

Conceptually:

```text
t-2             t-1              t
 │                │               │
 ▼                ▼               ▼
Input(t-2)      Input(t-1)      Input(t)
 │                │               │
 ▼                ▼               ▼
Hidden(t-2) ───► Hidden(t-1) ───► Hidden(t)
                                    │
                                    ▼
                                  Output
```

This allows the existing graph-based layer representation to model temporal dependencies without requiring an entirely separate RNN implementation.

The repository contains both standard and context-enabled weather models:

```text
weathrModel.cml
weathrModelContext.cml
weathrModelContextBest.cml
weathrPiModelBest.cml
```

---

# Model Serialization

Models can be stored as `.cml` files.

The CML interface is exposed through:

```text
src/include/cml.h
```

Current repository artifacts include:

```text
weathrModel.cml
weathrModelContext.cml
weathrModelContextBest.cml
weathrPiModelBest.cml
```

These files allow a trained model to be stored separately from the program executing it.

The deployment process can therefore be separated into:

```text
Training
   │
   ▼
CML model
   │
   ▼
Model loading
   │
   ▼
Inference
```

---

# Generated C Models

The repository also contains C representations of trained models:

```text
weatherPiModel.c
weatherPiModelContext.c
```

This provides another deployment path in which a model can be trained or developed on a more capable system and subsequently represented directly as C source code.

The intent is to make trained models easier to deploy into environments where loading the full training/model-construction pipeline is unnecessary.

---

# Weather Sensor Interface

The WeatherPi application currently interfaces directly with a **Bosch BME280** environmental sensor.

The sensor interface is implemented in:

```text
src/include/sensor/sensor.h
```

The implementation uses Linux's I²C device interface and communicates with:

```text
/dev/i2c-1
```

using the BME280 device address:

```text
0x76
```

The driver performs:

* Device identification
* Soft reset
* Calibration-data acquisition
* Temperature compensation
* Pressure compensation
* Humidity compensation
* Forced-mode measurements
* I²C register access

The sensor implementation also performs sea-level pressure conversion before passing the pressure value to the model.

---

# Weather Input Vector

The sensor interface produces a three-element input vector:

```text
[ temperature, pressure, humidity ]
```

`operate.c` allocates:

```c
float vals[3];
```

and passes this vector to `getWeatherInfo()`.

The resulting values are passed directly into recurrent model inference.

---

# WeatherPi Runtime

The current executable flow in `operate.c` is:

```text
Start
 │
 ▼
Load weathrModelContext.cml
 │
 ▼
Initialize/read BME280
 │
 ▼
Obtain temperature, pressure, humidity
 │
 ▼
Run rnn_model_inference()
 │
 ▼
Produce two output values
 │
 ▼
Print outputs
 │
 ▼
Destroy model
 │
 ▼
Exit
```

The actual application loads the context-enabled model, obtains weather measurements, calls `rnn_model_inference()`, prints the two outputs, and releases the model before exiting.

---

# Data Pipeline

The repository also contains weather datasets and preprocessing scripts.

The preprocessing pipeline is intended to transform raw weather measurements into model-ready training data.

The major dataset stages are:

```text
Raw weather data
      │
      ▼
Missing-data handling
      │
      ▼
Temporal differences
      │
      ▼
Normalization
      │
      ▼
Model-ready / encoded data
```

Relevant files include:

```text
dataset/thp.py
dataset/thpdiffs.py
dataset/thpdiffsnorm.py

dataset/THP.csv
dataset/THPDiffs.csv
dataset/THPDiffsNorm.csv
dataset/THPOneHot.csv
```

The resulting data is used for experimentation with precipitation prediction from environmental measurements.

---

# Training and Inference

The neural-network implementation is designed to support both training and inference.

The model stores training-specific state including:

```text
targets
lossDerivatives
learningRate
loss function
```

while each layer maintains intermediate values required for forward and backward computation:

```text
outputs
preActivations
activationDerivatives
backErrors
```

This allows the same underlying layer representation to participate in both inference and training.

---

# Core API Organization

The neural-network implementation is divided into several functional areas.

## Model

```text
src/include/model.h
src/include/model_construct.h
src/include/model_ops.h
src/include/model_destruct.h
```

Responsible for:

* Model representation
* Model construction
* Model destruction
* Model-level operations
* Layer ordering
* Input/output layer management
* Training state

## Layers

```text
src/include/layer.h
src/include/layer_construct.h
src/include/layer_destruct.h
```

Responsible for:

* Layer representation
* Layer construction
* Layer destruction
* Layer-level state

## Numerical Operations

```text
src/include/nn_math.h
src/include/nn_ops.h
src/include/helper_funcs.h
```

These provide the lower-level numerical and neural-network operations used by the model implementation.

## Activations

```text
src/include/activation.h
```

Contains activation functions and derivatives.

## Losses

```text
src/include/loss.h
```

Contains loss functions and derivatives.

## Serialization

```text
src/include/cml.h
```

Provides the CML model interface.

## Sensors

```text
src/include/sensor/sensor.h
```

Provides the BME280/Linux I²C interface.

---

# Testing

Test assets are located under:

```text
src/tests/
```

The repository includes model and dataset fixtures used to exercise model-loading and weather-model functionality.

The project also contains:

```text
src/tests/testWeatherModel.cml
src/tests/THPDiffsFull.csv
src/tests/THPOneHotFull.csv
```

These provide test inputs for the weather-model and data-processing paths.

---

# Current Capabilities

The current implementation provides:

* C-based neural-network representation
* Explicit layer and model memory management
* Graph-based layer connectivity
* Multiple predecessor layers
* Topologically ordered model execution
* Forward propagation
* Backpropagation infrastructure
* Configurable activation functions
* Configurable loss functions
* Glorot weight initialization
* Recurrent context windows
* Shared recurrent weights
* CML model serialization/loading
* Generated C model representations
* BME280 sensor integration
* Linux I²C communication
* Weather-data preprocessing
* Weather-model inference

---

# Future Development

The neural-network implementation is still under development.

Planned extensions include:

* Attention mechanisms for recurrent models
* Convolutional layers
* Multidimensional input tensors
* Image-oriented 3D inputs
* Video-oriented 4D inputs
* Higher-dimensional data representations
* Transformer-style self-attention
* Additional initialization strategies
* Further numerical optimization
* Additional embedded deployment functionality

---

# Design Philosophy

WeatherPi is intentionally implemented at a relatively low level.

Rather than treating the neural network as a black box provided by an external machine-learning framework, the project implements the core model representation, graph execution, activation functions, loss functions, memory management, recurrent context, serialization, and sensor interface directly in C.

The main design goals are:

* **Explicit memory management**
* **Flexible model topology**
* **Data-oriented execution**
* **Low-level numerical control**
* **Separation of model and layer lifecycles**
* **Recurrent model experimentation**
* **Portable serialized model artifacts**
* **Generated C deployment**
* **Direct hardware integration**

The project is therefore both a weather-prediction application and an experimental low-level neural-network framework.

---

# Hardware

The current weather application is designed around a BME280 environmental sensor connected through I²C.

The repository includes the sensor documentation:

```text
BME280 Sensor Manual.pdf
```

The current Linux implementation expects:

```text
I²C bus:       /dev/i2c-1
BME280 address: 0x76
```

The sensor provides the environmental measurements used by the model:

```text
Temperature
Pressure
Humidity
```

---

# License

Copyright © 2025 Tafadar Soujad.

This project is licensed under the **GNU General Public License v3.0**.

See [`LICENSE.txt`](LICENSE.txt) for the complete license text.https://github.com/trs-code/WeatherPi/tree/main
