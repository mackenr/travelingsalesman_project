# Finding solutions to the "Traveling Salesman Problem" via the genetic algorithim.


Richard Macken
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




















## Sample Data Dict of distances for n=36





|    | name   | data type   |   unique |   total count |   null count |   non null kurt |   percent null |
|---:|:-------|:------------|---------:|--------------:|-------------:|----------------:|---------------:|
|  0 | gen1   | float64     |       35 |            35 |            0 |     -0.272591   |              0 |
|  1 | gen10  | float64     |       35 |            35 |            0 |      0.109127   |              0 |
|  2 | gen100 | float64     |       35 |            35 |            0 |     -1.23742    |              0 |
|  3 | gen11  | float64     |       35 |            35 |            0 |     -0.739343   |              0 |
|  4 | gen12  | float64     |       35 |            35 |            0 |     -0.215712   |              0 |
|  5 | gen13  | float64     |       35 |            35 |            0 |     -0.575522   |              0 |
|  6 | gen14  | float64     |       35 |            35 |            0 |     -1.27945    |              0 |
|  7 | gen15  | float64     |       35 |            35 |            0 |     -0.707439   |              0 |
|  8 | gen16  | float64     |       35 |            35 |            0 |      1.42488    |              0 |
|  9 | gen17  | float64     |       35 |            35 |            0 |     -0.249124   |              0 |
| 10 | gen18  | float64     |       35 |            35 |            0 |     -1.05624    |              0 |
| 11 | gen19  | float64     |       35 |            35 |            0 |     -0.226803   |              0 |
| 12 | gen2   | float64     |       35 |            35 |            0 |      0.0582615  |              0 |
| 13 | gen20  | float64     |       35 |            35 |            0 |     -0.546488   |              0 |
| 14 | gen21  | float64     |       35 |            35 |            0 |      0.413751   |              0 |
| 15 | gen22  | float64     |       35 |            35 |            0 |      0.593468   |              0 |
| 16 | gen23  | float64     |       35 |            35 |            0 |      0.616037   |              0 |
| 17 | gen24  | float64     |       35 |            35 |            0 |      1.52429    |              0 |
| 18 | gen25  | float64     |       35 |            35 |            0 |     -0.626885   |              0 |
| 19 | gen26  | float64     |       35 |            35 |            0 |     -0.811789   |              0 |
| 20 | gen27  | float64     |       35 |            35 |            0 |     -0.853763   |              0 |
| 21 | gen28  | float64     |       35 |            35 |            0 |     -0.934949   |              0 |
| 22 | gen29  | float64     |       35 |            35 |            0 |     -0.481825   |              0 |
| 23 | gen3   | float64     |       35 |            35 |            0 |     -0.00411132 |              0 |
| 24 | gen30  | float64     |       35 |            35 |            0 |     -0.0493086  |              0 |
| 25 | gen31  | float64     |       35 |            35 |            0 |     -0.442681   |              0 |
| 26 | gen32  | float64     |       35 |            35 |            0 |     -0.186921   |              0 |
| 27 | gen33  | float64     |       35 |            35 |            0 |      0.310077   |              0 |
| 28 | gen34  | float64     |       35 |            35 |            0 |     -0.179506   |              0 |
| 29 | gen35  | float64     |       35 |            35 |            0 |     -0.173806   |              0 |
| 30 | gen36  | float64     |       35 |            35 |            0 |     -0.49656    |              0 |
| 31 | gen37  | float64     |       35 |            35 |            0 |     -0.959493   |              0 |
| 32 | gen38  | float64     |       35 |            35 |            0 |     -0.524657   |              0 |
| 33 | gen39  | float64     |       35 |            35 |            0 |     -0.687915   |              0 |
| 34 | gen4   | float64     |       35 |            35 |            0 |      0.0584625  |              0 |
| 35 | gen40  | float64     |       35 |            35 |            0 |     -0.589521   |              0 |
| 36 | gen41  | float64     |       35 |            35 |            0 |     -0.666479   |              0 |
| 37 | gen42  | float64     |       35 |            35 |            0 |     -0.235298   |              0 |
| 38 | gen43  | float64     |       35 |            35 |            0 |     -0.121617   |              0 |
| 39 | gen44  | float64     |       35 |            35 |            0 |     -0.701615   |              0 |
| 40 | gen45  | float64     |       35 |            35 |            0 |      0.00920138 |              0 |
| 41 | gen46  | float64     |       35 |            35 |            0 |     -0.284496   |              0 |
| 42 | gen47  | float64     |       35 |            35 |            0 |      0.144852   |              0 |
| 43 | gen48  | float64     |       35 |            35 |            0 |     -0.688837   |              0 |
| 44 | gen49  | float64     |       35 |            35 |            0 |     -0.506081   |              0 |
| 45 | gen5   | float64     |       35 |            35 |            0 |     -0.39513    |              0 |
| 46 | gen50  | float64     |       35 |            35 |            0 |      0.0474278  |              0 |
| 47 | gen51  | float64     |       35 |            35 |            0 |      0.16536    |              0 |
| 48 | gen52  | float64     |       35 |            35 |            0 |     -0.919171   |              0 |
| 49 | gen53  | float64     |       35 |            35 |            0 |     -0.290371   |              0 |
| 50 | gen54  | float64     |       35 |            35 |            0 |     -0.715303   |              0 |
| 51 | gen55  | float64     |       35 |            35 |            0 |      0.764635   |              0 |
| 52 | gen56  | float64     |       35 |            35 |            0 |     -0.770931   |              0 |
| 53 | gen57  | float64     |       35 |            35 |            0 |     -0.98007    |              0 |
| 54 | gen58  | float64     |       35 |            35 |            0 |     -0.538229   |              0 |
| 55 | gen59  | float64     |       35 |            35 |            0 |     -0.146443   |              0 |
| 56 | gen6   | float64     |       35 |            35 |            0 |     -0.424892   |              0 |
| 57 | gen60  | float64     |       35 |            35 |            0 |     -0.0173834  |              0 |
| 58 | gen61  | float64     |       35 |            35 |            0 |     -0.339321   |              0 |
| 59 | gen62  | float64     |       35 |            35 |            0 |     -0.309626   |              0 |
| 60 | gen63  | float64     |       35 |            35 |            0 |     -1.36806    |              0 |
| 61 | gen64  | float64     |       35 |            35 |            0 |     -0.972132   |              0 |
| 62 | gen65  | float64     |       35 |            35 |            0 |     -0.243613   |              0 |
| 63 | gen66  | float64     |       35 |            35 |            0 |     -0.460656   |              0 |
| 64 | gen67  | float64     |       35 |            35 |            0 |     -0.0375774  |              0 |
| 65 | gen68  | float64     |       35 |            35 |            0 |     -0.275685   |              0 |
| 66 | gen69  | float64     |       35 |            35 |            0 |     -0.746702   |              0 |
| 67 | gen7   | float64     |       35 |            35 |            0 |      0.407866   |              0 |
| 68 | gen70  | float64     |       35 |            35 |            0 |     -0.243473   |              0 |
| 69 | gen71  | float64     |       35 |            35 |            0 |     -0.862587   |              0 |
| 70 | gen72  | float64     |       35 |            35 |            0 |     -0.25941    |              0 |
| 71 | gen73  | float64     |       35 |            35 |            0 |     -0.849819   |              0 |
| 72 | gen74  | float64     |       35 |            35 |            0 |     -0.517049   |              0 |
| 73 | gen75  | float64     |       35 |            35 |            0 |     -0.566755   |              0 |
| 74 | gen76  | float64     |       35 |            35 |            0 |     -1.14403    |              0 |
| 75 | gen77  | float64     |       35 |            35 |            0 |     -0.49474    |              0 |
| 76 | gen78  | float64     |       35 |            35 |            0 |     -0.574293   |              0 |
| 77 | gen79  | float64     |       35 |            35 |            0 |     -0.146946   |              0 |
| 78 | gen8   | float64     |       35 |            35 |            0 |     -0.113747   |              0 |
| 79 | gen80  | float64     |       35 |            35 |            0 |      0.0270132  |              0 |
| 80 | gen81  | float64     |       35 |            35 |            0 |     -0.572362   |              0 |
| 81 | gen82  | float64     |       35 |            35 |            0 |     -0.223018   |              0 |
| 82 | gen83  | float64     |       35 |            35 |            0 |     -0.266947   |              0 |
| 83 | gen84  | float64     |       35 |            35 |            0 |     -1.06077    |              0 |
| 84 | gen85  | float64     |       35 |            35 |            0 |     -0.440283   |              0 |
| 85 | gen86  | float64     |       35 |            35 |            0 |     -0.834024   |              0 |
| 86 | gen87  | float64     |       35 |            35 |            0 |     -0.314972   |              0 |
| 87 | gen88  | float64     |       35 |            35 |            0 |     -0.397636   |              0 |
| 88 | gen89  | float64     |       35 |            35 |            0 |     -0.423457   |              0 |
| 89 | gen9   | float64     |       35 |            35 |            0 |     -0.510499   |              0 |
| 90 | gen90  | float64     |       35 |            35 |            0 |     -1.22258    |              0 |
| 91 | gen91  | float64     |       35 |            35 |            0 |      0.318598   |              0 |
| 92 | gen92  | float64     |       35 |            35 |            0 |     -0.744748   |              0 |
| 93 | gen93  | float64     |       35 |            35 |            0 |     -0.588594   |              0 |
| 94 | gen94  | float64     |       35 |            35 |            0 |     -0.628682   |              0 |
| 95 | gen95  | float64     |       35 |            35 |            0 |     -0.77224    |              0 |
| 96 | gen96  | float64     |       35 |            35 |            0 |      1.33999    |              0 |
| 97 | gen97  | float64     |       35 |            35 |            0 |     -0.464738   |              0 |
| 98 | gen98  | float64     |       35 |            35 |            0 |     -0.723187   |              0 |
| 99 | gen99  | float64     |       35 |            35 |            0 |     -1.066      |              0 |



















