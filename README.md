
<img src=https://see.fontimg.com/api/renderfont4/KpAp/eyJyIjoiZnMiLCJoIjo1NywidyI6MTAwMCwiZnMiOjU3LCJmZ2MiOiIjMzFBMEVCIiwiYmdjIjoiI0ZGRkZGRiIsInQiOjF9/QXV0b1NvdXJjZUlELUNsYXNzaWZpY2F0aW9u/kg-second-chances-sketch.png>

## Overview
ASID-C is an algorithm for the classification of stars and galaxies in single band optical images. 
ASID-C uses cutouts of 32x32 pixels of sources localized by AutoSourceID-Light (ASID-L) and assigns a calibrated probability to belong to either class.
Multiple Deep Learning networks have been applied and compared, leading to a rather surprising result: a vanilla convolutional neural network (CNN) is the best suited for the task.

## Table of Contents 
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [License](#license)

## Installation

1. Clone the repo
   ```sh
   git clone https://github.com/FiorenSt/AutoSourceID-Classifier.git
   ```

# Dependencies:

* Python 3.7 (or superior)
* TensorFlow 2 
* Scikit-Image 0.18.1
* Numpy 1.20.3
* Astropy 4.2.1
* glob2

You can install them using pip:

```bash
pip install tensorflow==2.* scikit-image==0.18.1 numpy==1.20.3 astropy==4.2.1 glob2
```

This combination of package versions works on most Linux and Windows computers, however other package versions may also work.
If the problem persists, raise an issue and we will help you solve it.

## Data

The data for this project is available on Zenodo. To download and prepare the data, follow these steps:

1. Download the data from Zenodo.

2. Unzip the downloaded file. This will give you four folders: `Training`, `Test`, `Validation`, and `Calibration`.

3. Copy these four folders into the `Data` folder in the project directory. The structure should look like this:

```text
Project Folder
|
|--- Data
     |
     |--- Training
     |--- Test
     |--- Validation
     |--- Calibration
```

## Usage

The use of the pre-trained ASID-C is straight forward: 

```bash
python main.py
```

By default, this will load the pre-trained model and use it to make predictions on the test data. If you want to train a new model, you can modify the `main.py` script accordingly.

## License

Copyright 2023 Fiorenzo Stoppa

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
