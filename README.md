# Project Title

Transformations of the spatial activity manifold convey aversive information in CA3

## Description

Scripts to create the figures in the paper "Transformations of the spatial activity manifold convey aversive information in CA3".

The necessary data is from the paper "Efficient encoding of aversive location by CA3 long-range projections"
Ref: 
    Efficient encoding of aversive location by CA3 long-range projections
    Nikbakht, Negar et al.
    Cell Reports, Volume 43, Issue 3, 113957


## Getting Started

### Dependencies

* Python 3.9
* Poetry for Python to install the required dependencies
  * Installation instructions: https://python-poetry.org/docs/
* A Python environment manager such as Conda is highly recommended

### Installing

* Create a new empty environment through an environment manager such as Conda
* Open a terminal and move to the locally cloned repository folder
* Run "poetry install" to install all dependencies
  * If it doesn't work, make sure you are in the folder that has a ".toml" file
* Create a folder for the project and add it to the "recognized_project_paths" parameter in "project_parameters.py"
* Add the data array to the folder and make sure the "FAT_CLUSTER_PATH" parameter in "project_parameters.py" is correct
* [Only for figure 5] Add the secondary data array ("place_cell_bool_dombeck.mat") and make sure the "PLACE_CELL_PATH_DOMBECK" parameter in "project_parameters.py" is correct
                      

### Executing program

* Open "main_figures.py" in a code editor
* Comment / Uncomment the figures and subfigures you want to run in the "main()" function
* Run "main_figures.py"

## Authors

For any issues or questions, please contact:

Albert Miguel López
Current work mail: amiguel@uni-bonn.de
Personal mail: amiguello2@gmail.com
