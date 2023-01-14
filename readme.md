# Traveling Salesman Problem:
## Solutions via the genetic algorithim.

### -Richard Macken

 <img alt="Richard" src="https://img.shields.io/github/followers/mackenr?label=Follow Rich&style=social" />
<img alt="Example" src=Code_Up_TSP_Simulation_n24.png />

---

> Note: This an interactive notebook. In order to obtain the entire funcionality you will need to clone the repo, install dependances and run it. If you notice an unfamilar import you will need to install it via your package manager of choice.
```py 
from __future__ import division
import string
from itertools import chain
import os
import sys
from time import sleep
import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.graph_objects as go
from IPython.display import Image,display, clear_output
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from matplotlib import collections  as mc
from itertools import permutations,combinations
from matplotlib_scalebar.scalebar import ScaleBar
import time
import seaborn as sns
from scipy.special import gamma
import plotly.express as px
import geopandas as gpd
import shapely
import numpy as np
from sympy import symbols
import wget
from math import radians, cos, sin, asin, sqrt, factorial
import scipy.stats as stats
```

---

---- 
## About:
This was a base case to understand an algorithm that I had not used before. In this specific implementation it actually solves a simply stated but deceptively complex problem, which is to find the shortest path between n points on a map. A real world application could be an oil and gas company who wanted to find the optimal path to fly a drone to hit every construction site within a certain radius.

I used this project as a first  step towards understanding genetic algorithms and exploring geographical data. The benefit is that these algorithms allow you to move forward when finding the correct answer is dependent on exploring every possible solution. Finding and exploring the entire solutions’ permutations set can become so computationally expensive that it is not possible to calculate the answer within a reasonable amount of time.  I used a classic problem in applied mathematics (Traveling Salesman Problem) as an example where I use an implementation of the genetic algorithm to find a solution. The current implementation is a base case and simplified. It simply finds the shortest distance between n points on a map. However as it currently stands it could be used to minimize the distance  a drone flies given a set of geographical points. In the future, I would like to go further and consider a related problem called the multi vehicle routing problem. This problem finds the answer to how many vehicles are required to minimize the time-cost of hitting all the locations.

Additionally this problem helps a junior data professional by framing some of the abstractions underlying clustering and k-nearest neighbors as real world problems. If one develops a working knowledge of creating genetic algorithms, they can use them to tune areas of the data science pipeline by improving models which already exist or giving starting points to problems that are otherwise unmanageable. 



## What is the use?

Well the TSP is a classic problem in applied mathematics. With little modification it can be applied to drones. For a broader customer base, it is foundational to understanding and solving more complex problems such as fleet optimization for any shipping and delivery system. This class of problem is often called a vehicle routing problem.

Yet since it is asking for the shortest path connecting nodes on a graph, it is a great base case to give us a fundamental understanding of many graph and decision tree based models.
The naive approach is to consider the permutations of the problem which as we showed above can be computationally expensive to the point it becomes impossible when our number of nodes becomes large.

Here we leverage a metaheuristic called a genetic algorithm. This gives a near optimal answer to our problem. It can be used in many applications as mentioned earlier. Many financial engineers use it when analyzing financial models
So is a genetic algorithm a machine learning model in itself? Well it depends on who you ask, there are those who say yes and those who say no. What is clear is that it is a useful tool for Data scientists to use and understand as it can allow us to optimize our staple machine learning models.

In short, our exploration offers solutions to specific real world problems while establishing a level of clarification to the mechanisms underlying many machine learning models.


---
## Wrangle:

Functions:
The set of latitude and longitudes generated were created by a function I made. It uses the coordinates for Code Up and it then creates n random coordinates restricted by a radius.

Modules used:

For this project the data was randomly created. Initially in order to make this relevant I attempted to source data from my fellow classmates and alumni where they would give approximate coordinates to a near to where they lived while attending code up. However, in order to avoid the appearance of any personally identifiable information being published I choose random values instead.

---

## Anatomy of a genetic algorithm:

#### Initialization:
You need an initial population. In our case it is an ordered list of  geo-coordinates.
Fitness assessment
You need a way to assess fitness. This is what makes the algorithm so dynamic. The designer needs to put some thought into this step as it is the most fundamental part of the process.

#### Selection
You need a way to select those that will pass on their genes

#### Crossover
You need to implement the actual crossover
Mutation
A small degree of mutation has to happen. This is random and typically set to something like 1 percent.

#### Stopping criteria:

In this case it was simply the amount of generations, you can use other stopping criteria though.



---


## Data split:

For this project I saw no need to split the data as the tool here is answering a specific problem respective to coordinates given. There’s no risk of skewing our data since this is an optimization model that makes a probabilistic attempt at finding the minimum distance. 

---
## Encoding of categorical variables:

The encoding happens at a low level but it is fundamental to the genetic algorithm. In each instance a unique  (x=long,y=lat) is mapped to a unique number. The order of those numbers are swapped to give 100 permutations each generation.

## Scaling:
The only scaling that happened was simply to ensure everything was in the same units. For the sake of ease they are all metric units. In order to generate distances we converted the lat and long degrees into kilometers. However the distance function I used to compare later on is an abstraction. It is a euclidean distance of the lat and long. I did not convert this into kilometers because when I attempted python was interpreting some of the values as zeros or infinity. The abstract distance instead seemed to be a reasonable approximation since the distances were so close.


---

## Explore:

There was a lot to reverse engineer. I will not say I have 100% confidence in this current iteration. I was able to get the algorithm to run by pulling from several sources online. I attempted to extract information at key points to make statistical comparisons. 
Reproducibility:

The generator function allows one to recreate the process. If I was a bit more clever in my reverse engineering we would be able to recreate more aspects on a 1 to 1 basis however this algorithm leverages random mutation rates everytime. To make every single metric reproducible is challenging. The net process is reproducible, the exact numbers are going to vary in the current implementation.

We focused on a specific model to explore but I made several. In the mainshow() there is a quick montage that goes through all the models.

I tried my best to apply the data science pipeline to this project but it was a different application, one that is more of an assistance to the staple data science pipelines yet can be standalone. 

When we observe all the models we get to see other factors that are important for algorithm analysis such as the time it took to run each model. The most important takeaway at the individual scale is the  visualization of the distance being minimized but then fluctuating as the genetic algorithm does not guarantee convergence. So we see the pros and cons of the algorithm.




The stats were not as meaningful as I would like simply because I really don't have expereince evaluating algorithims.

---

## Lessons Learned:

There was a ton to unpack. I understand this at a high level. I was able to reverse engineer the algorithm by copying some code from various sources. Then I had to make a new version with an implementation that took actual lat and long values. However, my stats testing was not as informative as I wanted because there are 750 different generations and I did not get to the point where I could grab each generation. It is something I will have to revisit at a later date.

---

# Executive Summary:

I was successful at implementing the genetic algorithm which picks a minimal tour between a set of geographical coordinates. That said, there are still steps in this process which are abstracted in the current implementation. 

Future goals are to uncover a bit more of the abstraction, in that way I will be able to utilize this tool more dynamically as it is intended. I also want to implement it to solve a related problem which is the multi vehicle routing problem. This is more useful as it can optimize fleets of vehicles and be used to save firms money. 

I also intend to use this algorithm to optimize staple machine learning models such as knn and hyper parameters for others.

























