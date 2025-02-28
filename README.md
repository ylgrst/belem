# Belem

Belem is an open-source Python library to be used in conjuction with [3MAH's suite of FEA tools](https://github.com/3MAH).
More specifically, it is compatible with Simcoon 1.9.6 and Fedoo 0.2.3.
Belem is designed as a tool for non-linear homogenization of mechanical properties of architectured materials.

## Contents

### fem module

The fem module contains classes and methods necessary for FEA computation, postprocessing, homogenized material law identification and report generation.

### latticegen module

The latticegen module allows for generation of lattice-based architectured cells CAD and meshing. The implemented lattice structures are Body Centered Cubic, Cuboctahedron, Kelvin, Octet Truss and Truncated Octahedron.

### scripts

This folder contains examples showing how to use this library for a single FEA computation, for a full linear homogenization analysis, and well as a complete non-linear homogenization process for one type of architectured unit-cell.

## Installation

Simply clone this repo, cd into it then install using pip:
```
git clone https://github.com/ylgrst/belem.git
cd belem
pip install .
```
[!NOTE] 
Parts (if not all) of this repo will be integrated to 3MAH softwares in the future
