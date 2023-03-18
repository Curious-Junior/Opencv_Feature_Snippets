# Using the source files to extract features form your own data.

## Pre-requisites

Install python, pip and the opencv-python package.

###In debian-based linux distros:

sudo apt-get update
sudo apt-get install python3
sudo pip install opencv-python

And use apt equivalents in other systems.

###On Windows:

Install python form python's website.
Run pip install opencv-python in cmd or powershell

### Instructions not available for mac

## SURF usage

Install and build opencv and opencv-contrib from source.
Both sources are available on github.

In CMAKE, make sure to set the EXTRA_MODULES_PATH to the <path to opencv-contrib>/modules and ENABLE_NONFREE to ON.
Now, add the built library to your PYTHONPATH

Note: this process can be really problematic.

## The easiest way to use source files

Go into Source Files/featuresAndCornersDiagrams.py and replace the file path in line 10 with your own file path.

## More customizable option

Import the featuresAndCornersDiagrams and featureAndCornersDetectors files into your own python file.
Use each of the feature detection features with your own arguments(definitions are in docstrings)
