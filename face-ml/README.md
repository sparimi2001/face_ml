# Pocofy Face Recognition and Gender Classification
## Overview
The program **face_ml.py** takes a path of a directory with image with an unknown face and a path to a gallery of images with known faces as inputs through command line arguments.

Each image must have only one face.

The program will compare the unknown face with the gallery and will return the corresponding file name if the unknown face matches with a known face.

The program will also identify the unknown face's gender.

## Installation
You need to install the [face_recognition module] (https://github.com/ageitgey/face_recognition#installation). The instructions are below:

### Requirements

* Python 3.3+ or Python 2.7
* macOS or Linux (Windows not officially supported, but might work)

First, make sure you have dlib already installed with Python bindings:

* [How to install dlib from source on macOS or Ubuntu](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)

Then, install this module from pypi using pip3 (or pip2 for Python 2):

	pip3 install face_recognition

## Example Usage
* Change to the directory containing the program
* Run the program using python3

		$ cd Final_Program
		$ python3 face_ml.py
		Using TensorFlow backend.
		2018-07-16 20:52:59.173000: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
			
		Angelina_Jolie_0002.jpg is being analyzed
		[ INFO:0] Initialize OpenCL runtime...
		match from gallery: Angelina_Jolie_0001.jpg
		gender: woman

		Angelina_Jolie_0020.jpg is being analyzed
		match from gallery: Angelina_Jolie_0001.jpg
		gender: woman
		
		Anna_Kournikova_0002.jpg is being analyzed
		match from gallery: Anna_Kournikova_0001.jpg
		gender: man
		
		Anna_Kournikova_0003.jpg is being analyzed
		Match not found
		gender: woman
		
		Bill_Gates_0002.jpg is being analyzed
		match from gallery: Bill_Gates_0001.jpg
		gender: man
		
		Bill_Gates_0003.jpg is being analyzed
		Error: More than one face found
		
		Bill_Gates_0004.jpg is being analyzed
		match from gallery: Bill_Gates_0001.jpg
		gender: man
		
		...
		
		$