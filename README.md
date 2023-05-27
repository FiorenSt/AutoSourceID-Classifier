<img src=https://see.fontimg.com/api/renderfont4/KpAp/eyJyIjoiZnMiLCJoIjo1NywidyI6MTAwMCwiZnMiOjU3LCJmZ2MiOiIjMzFBMEVCIiwiYmdjIjoiI0ZGRkZGRiIsInQiOjF9/QXV0b1NvdXJjZUlELUNsYXNzaWZpY2F0aW9u/kg-second-chances-sketch.png>


<!--
<img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/OpticalImagePatch.png " width=50% height=50%><img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/LoGOnOptical.png " width=50% height=50%> 
-->


## Overview
ASID-C is an algorithm for the classification of stars and galaxies in single band optical images. 
ASID-C uses cutouts of 32x32 pixels of sources localized by AutoSourceID-Light (ASID-L) and assignes a calibrated probability to belong to either class.
Multiple Deep Learning networks have been applied and compared, leading to a rather surpirsing result: a vanilla convolutional neural network (CNN) is the best suited for the task.


## Table of Contents 
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)


## Installation

1. Clone the repo
   ```sh
   git clone https://github.com/FiorenSt/AutoSourceID-Classifier.git
   ```

# Dependencies:

* Python 3 (or superior)
* TensorFlow 2 
* Scikit-Image 0.18.1
* Numpy 1.20.3
* Astropy 4.2.1

This combination of package versions works on most Linux and Windows computers, however other package versions may also work.
If the problem persist, raise an issue and we will help you solve it.



## Usage

The use of the pre-trained ASID-C is straight forward: 

```
python main.py
```

It loads a .fits image from the Data folder and the pre-trained model, and it outputs, for each source, the probability of it being a star.

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







