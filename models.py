
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


def generator():
    ''' This is what you must run to aquire data. The nature of the algorithm is going to create differeing results everytime becasue of the mutation rates. This allows you to recreate the process
    '''

    listtoindex=list(range(3,12,1))
    listtoadd=list(range(12,66,6))
    listtoindex.extend(listtoadd)
   
    

    for i in listtoindex:

        a=geneticAlgorithm_with_Plot(n=i,popSize=100, eliteSize=20, mutationRate=0.01, generations=int(750),show=False)
        bestRoutedist(a,i)
       
    
    



codeupmaps=[
'Code_Up_TSP_Simulation_n3.png',
'Code_Up_TSP_Simulation_n4.png',
'Code_Up_TSP_Simulation_n5.png',
'Code_Up_TSP_Simulation_n6.png',
'Code_Up_TSP_Simulation_n7.png',
'Code_Up_TSP_Simulation_n8.png',
'Code_Up_TSP_Simulation_n9.png',
'Code_Up_TSP_Simulation_n10.png',
'Code_Up_TSP_Simulation_n11.png',
'Code_Up_TSP_Simulation_n12.png',
'Code_Up_TSP_Simulation_n18.png',
'Code_Up_TSP_Simulation_n24.png',
'Code_Up_TSP_Simulation_n30.png',
'Code_Up_TSP_Simulation_n36.png',
'Code_Up_TSP_Simulation_n42.png',
'Code_Up_TSP_Simulation_n48.png',
'Code_Up_TSP_Simulation_n54.png',
'Code_Up_TSP_Simulation_n60.png']


geneticAlgorithmperfomancegraphs=[
'GeneticAlgoPlot_TSP_3.png',
'GeneticAlgoPlot_TSP_4.png',
'GeneticAlgoPlot_TSP_5.png',
'GeneticAlgoPlot_TSP_6.png',
'GeneticAlgoPlot_TSP_7.png',
'GeneticAlgoPlot_TSP_8.png',
'GeneticAlgoPlot_TSP_9.png',
'GeneticAlgoPlot_TSP_10.png',
'GeneticAlgoPlot_TSP_11.png',
'GeneticAlgoPlot_TSP_12.png',
'GeneticAlgoPlot_TSP_18.png',
'GeneticAlgoPlot_TSP_24.png',
'GeneticAlgoPlot_TSP_30.png',
'GeneticAlgoPlot_TSP_36.png',
'GeneticAlgoPlot_TSP_42.png',
'GeneticAlgoPlot_TSP_48.png',
'GeneticAlgoPlot_TSP_54.png',
'GeneticAlgoPlot_TSP_60.png']
images=[]
# images = list(chain(*zip(codeupmaps,geneticAlgorithmperfomancegraphs)))
images.extend(codeupmaps)
images.extend(geneticAlgorithmperfomancegraphs)
indexs=[i for i in range(len([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 24, 30, 36, 42, 48, 54, 60]))]
dictforindexs=dict(zip([3, 4, 5, 6, 7, 8, 9, 10, 11, 12,18, 24, 30, 36, 42, 48, 54, 60],indexs))



def TSPresultsfastViz():
    s=.15
    print("Montage ")
    sleep(.75)
    clear_output()
    sleep(.5)
   
    for image in images:
          
        display(Image(filename=image))
        sleep(s)
        clear_output()

def TSPFocusedViz(n):
    n=dictforindexs.get(n)

    display(Image(filename=codeupmaps[n]))
    display(Image(filename=geneticAlgorithmperfomancegraphs[n]))










def prRed(skk): print("\033[91m {}\033[00m" .format(skk))
 
 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))
 
 
def prYellow(skk): print("\033[93m {}\033[00m" .format(skk))
 
 
def prLightPurple(skk): print("\033[94m {}\033[00m" .format(skk))
 
 
def prPurple(skk): print("\033[95m {}\033[00m" .format(skk))
 
 
def prCyan(skk): print("\033[96m {}\033[00m" .format(skk))
 
 
def prLightGray(skk): print("\033[97m {}\033[00m" .format(skk))
 
 
def prBlack(skk): print("\033[98m {}\033[00m" .format(skk))





def f1(x):
    return (0*x)+1
label1= '$O(1)$'

def f2(x):
    return np.log(x) 
label2= '$O(log n)$'

def f3(x):
    return x
label3= '$O(n)$'

def f4(x):
    return  x*np.log(x) 
label4= '$O(n \\ log \\ n)$'

def f5(x):
    return x**2
label5= '$O(n^2)$'

def f6(x):
    return 2**x
label6= '$O(2^n)$'


label7= '$O(n!)$'

def f8(x):
    return x**x
label8= '$O(n^n)$'

def bigOplot(xmax=15):
    x=np.linspace(0,xmax*1.25,1000)
    x1=np.linspace(1,xmax*1.25,1000)
    ymax=(xmax**xmax)+xmax*xmax
    
    fig = plt.figure(figsize=(5,5)) 
    ax=fig.add_subplot(1,1,1)


    ax.plot(x,f1(x),'lime',label=label1)
    ax.plot(x,f2(x1),'peru',label=label2)
    ax.plot(x,f3(x),'crimson',label=label3)
    ax.plot(x,f4(x1),'gold',label=label4)
    ax.plot(x,f5(x),'navy',label=label5)
    ax.plot(x,f6(x),'c',label=label6)
    ax.plot(x,gamma(x),'w',label=label7)
    ax.plot(x,f8(x),'purple',label=label8)
    ax.set_title('$Big$ $O$ $Notation$')
    ax.set_xlabel('$Elements$')
    ax.set_ylabel('$Operations$')
    ax.legend(loc='upper right')
    ax.set_ylim(0,ymax)
    return plt.show()



def bigOfactplot(xmax=100):
    x=np.linspace(0,xmax*1.25,100000)
    ymax=float(gamma(xmax+1)+10)
  
    
    fig = plt.figure(figsize=(7,7)) 
    ax=fig.add_subplot(1,1,1)


   
    ax.plot(x,gamma(x),'crimson',label=label7)
  
    ax.set_title('$Big$ $O$ $Notation$')
    ax.set_xlabel('$Elements$')
    ax.set_ylabel('$Operations$')
    ax.legend(loc='upper right')
    ax.set_ylim(0,ymax)
    ax.set_xlim(0,xmax+15)
    return plt.show()

def bigObigscale():
    print('Small scale big O plot\n. ')
    bigOplot(xmax=10)
    sleep(1.75)
    clear_output()
    # print('Larger Scale \n')
    # bigOplot(xmax=20)
    # sleep(1.75)
    # clear_output()
    print('To highlight the our problem we look at f(60) when f(n)=n! \n We see at this larger scale we see magnitude of O(n!) as its curve becomes virtually asymptotic!.')
    bigOfactplot(xmax=60)
   






def showfactorials():
    n=list(range(0,125,5))
    



    intervals='Let us begin. Observe how aggressively the rate O(n!) increases \nWe compute f(n!) from 0 to 125 at intervals of 5:\n'
   


    for char in intervals:
            sleep(0.009)
            print(char, end='', flush=True)

    sleep(3)
    clear_output()

   
    countdown=['3\n','2\n','1\n','GO!!!\n\n\n']
    

    for char in countdown:
        sleep(0.15)
        print(char, end='', flush=False)
        sleep(0.15)

    sleep(.25)

    clear_output()
    print('\n\n')
      



    for i in n:
        h=f'{factorial(i)}\n'
        
        for char in h:
            sleep(0.0001)
            print(char, end='', flush=True)
            
    sleep(1.5)
    clear_output()
    sleep(.5)


    highligthyper='\033[92m "optimizing decision trees and hyperparameter optimization"'
            
    motivation=f'''
    First we consider the limitations of time complexity recall "big O" notation O(n):
    
    Solving things by "brute force" is often impossible or at least impractical. Many problems can be
    reduced to permutations however permutations are of the order O(n!).
    We will find a way around this by exploring a decievingly simply problem,a classic problem in mathematics called the traveling salesman problem (TSP).    

    From the wikipedia article:

    "An exact solution for 15,112 German towns from TSPLIB was found in 2001 using the cutting-plane method proposed by George Dantzig,
    Ray Fulkerson, and Selmer M. Johnson in 1954, based on linear programming. The computations were performed on a network of 110 processors located
    at Rice University and Princeton University. The total computation time was equivalent to 22.6 years on a single 500 MHz Alpha processor.
    In May 2004, the travelling salesman problem of visiting all 24,978 towns in Sweden was solved: a tour of length approximately 72,500 kilometres
    was found and it was proven that no shorter tour exists.[26] 
    In March 2005, the travelling salesman problem of visiting all 33,810 points in a circuit board was solved using
    Concorde TSP Solver: a tour of length 66,048,945 units was found and it was proven that no shorter tour exists. 
    The computation took approximately 15.7 CPU-years (Cook et al. 2006). In April 2006 an instance with 85,900 points was solved using Concorde TSP Solver, 
    taking over 136 CPU-years, see Applegate et al. (2006)."
    

   To solve our far simpler TSP, will be using the so called "genetic algorithm". This is based directly on Dawin's theory of evolution.
   This evolution is both a strength and weakness of the algorithm
   
   In short this tool is very dynamic and faster than many "brute force" methods. It will give us a "good enough" answer which is useful
   when turn around time is short or when you need or  simply to get a general direction to focus on when you are otherwise lost in a complex problem. 



    From the wikipedia article:

    "In computer science and operations research, a genetic algorithm (GA) is a metaheuristic inspired by the process of natural selection that belongs 
    to the larger class of evolutionary algorithms (EA). Genetic algorithms are commonly used to generate high-quality solutions to optimization and search problems 
    by relying on biologically inspired operators such as mutation, crossover and selection.[1] Some examples of GA applications include optimizing decision trees for better performance, 
    solving sudoku puzzles,[2] hyperparameter optimization, etc."
    
    The sharp eyed among you will note a common use is {highligthyper}
    
    '''

    
    s=.009
    lastwords=(' - a very interesting tool indeed.')

    for char in motivation:
           
            sleep(s)
            print(char, end='', flush=True)

    
    sleep(.75)

    prLightGray('')

    sleep(.75)

    for char in lastwords:
           
            sleep(s)
            print(char, end='', flush=True)
           

def theMainShow(n):
    '''
    The n is the number of nodes to focus on 
    '''

    slidebreaks()     
    showfactorials()
    slidebreaks()
    bigObigscale()   
    TSPresultsfastViz()
    slidebreaks()
    TSPFocusedViz(n)
    slidebreaks()
    print('\nThank you\n') 



def slidebreaks():
    '''
    I used this a a few timnes it creates a prompt to proceed before moving on to the next portion.
    
    '''
    sleep(.5) 
 
    print('\n\n-[Press any key to proceed]-\n')
    x = input()
    clear_output()
    sleep(.5)   

    

  


  
    
   
    
   
   
   
   




##### The models code (genetic alg ) code #############

















class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"



##Wrangle

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r









def nearoptimalcircuit(bestroute):
    fig = plt.figure(figsize=(15,15))
    ax = host_subplot(111, axes_class=AA.Axes)
   

     
    latitude1,longitude1 = 29.426976974865532, -98.48955221418564 #codeup

    ax.plot(longitude1,latitude1,marker='$C$',color="#77AC30")

    lines =[]
    path=[] 
    x=[]
    y=[]
    path=[]  


    rangeforloop=list(range(0,len(bestroute)-1))
    if len(bestroute)%2!=0:
        for i in rangeforloop:
            subline=[]
            if i==1:
                x1=bestroute[i+1].x
                y1=bestroute[i+1].y
                x2=bestroute[i].x
                y2=bestroute[i].y
                ax.plot(x1,y1,color="green", marker="o")
                ax.plot([x1,x2],[y1,y2],color="green",   linestyle=":")
            elif i==rangeforloop[-1]:
                x1=bestroute[i+1].x
                y1=bestroute[i+1].y
                x2=bestroute[i].x
                y2=bestroute[i].y
                x3=bestroute[0].x
                y3=bestroute[0].y

                ax.plot([x1,x2],[y1,y2],color="white",marker='o' ,   linestyle=":")
                ax.plot([x2,x3],[y2,y3],color="red",   linestyle=":")
                ax.plot(x1,y1,color="blue")
                ax.plot(x3,y3,color="red", marker="x")

            elif i%2==0:               
                x1=bestroute[i+1].x
                y1=bestroute[i+1].y
                x2=bestroute[i].x
                y2=bestroute[i].y
                ax.plot([x1,x2],[y1,y2],color="white",marker='o'  ,linestyle=":")
                # x.append(x1)
                # y.append(y1)
                # path.append(i)
            else:
                x1=bestroute[i+1].x
                y1=bestroute[i+1].y
                x2=bestroute[i].x
                y2=bestroute[i].y
                ax.plot([x1,x2],[y1,y2],color="blue",  linestyle=":")
                # x.append(x1)
                # y.append(y1)
                # path.append(i)
    else:
        for i in rangeforloop:
            subline=[]
            if i==1:
                x1=bestroute[i+1].x
                y1=bestroute[i+1].y
                x2=bestroute[i].x
                y2=bestroute[i].y
                ax.plot(x1,y1,color="green", marker="o")
                ax.plot([x1,x2],[y1,y2],color="green",   linestyle=":")
            elif i==rangeforloop[-1]:
                x1=bestroute[i+1].x
                y1=bestroute[i+1].y
                x2=bestroute[i].x
                y2=bestroute[i].y
                x3=bestroute[0].x
                y3=bestroute[0].y

                ax.plot([x1,x2],[y1,y2],color="white",marker='o' ,   linestyle=":")
                ax.plot([x2,x3],[y2,y3],color="red",   linestyle=":")
                ax.plot(x1,y1,color="blue")
                ax.plot(x3,y3,color="red", marker="x")

            elif i%2!=0:               
                x1=bestroute[i+1].x
                y1=bestroute[i+1].y
                x2=bestroute[i].x
                y2=bestroute[i].y
                ax.plot([x1,x2],[y1,y2],color="white",marker='o'  ,linestyle=":")
                # x.append(x1)
                # y.append(y1)
                # path.append(i)
            else:
                x1=bestroute[i+1].x
                y1=bestroute[i+1].y
                x2=bestroute[i].x
                y2=bestroute[i].y
                ax.plot([x1,x2],[y1,y2],color="blue",  linestyle=":")       

            # subline.append((x1,y1))
            # subline.append((x2,y2))
            # lines.append(subline)
    # points=list(zip(x,y))
    # viridis = mpl.colormaps['viridis']
    # viridis=viridis(np.linspace(0, 1, len(lines)))
    # lc = mc.LineCollection(lines, colors=viridis, linewidths=2)
    # ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    plt.show()

    # print(f'If we were to solve this by brute force we would need to explore all our permutations.\nThis would be a total of {factorial(len(bestroute)):.2g} explorations')

    


def create_random_point(x0,y0,distance):
    """
           This is how we generate our data for simulation of the points using codeup as our centeral location. The geopcord inates are random but based on a radious in kilometers
           from a central point which is the lat and long of Code Up
    """   
    r = distance/(111139)
    u = np.random.uniform(0,1)
    v = np.random.uniform(0,1)
    w = r * np.sqrt(u)
    t = 2 * np.pi * v
    x = w * np.cos(t)
    x1 = x / np.cos(y0)
    y = w * np.sin(t)
    return (x0+x1, y0 +y)





def latlonglist(n,show):
    '''
    
    creates our intital locations this is how we aquire our data
    
    '''
    print (f"Our lat and longs: \n")    # a value approxiamtely less than 150 km 
    fig = plt.figure(figsize=(15,15))
    ax = host_subplot(111, axes_class=AA.Axes)

    #ax.set_ylim(76,78)
    #ax.set_xlim(13,13.1)
    ax.set_autoscale_on(True)

    latitude1,longitude1 = 29.426976974865532, -98.48955221418564 #codeup
    ax.plot(longitude1,latitude1,marker='$C$',color='#77AC30')
    latlonglist=[]
 

    for i in range(1,n+1):
       
        x,y = create_random_point(longitude1,latitude1 ,int(80*1e3) )
        latlonglist.append( City(x=x,y=y))
        ax.plot(x,y,'bo')
       
        
        # print (f"Distance between points ({x},{y}is:\n{dist}")    # a value approxiamtely less than 150 km 
    if show==True:
        plt.show()
    
   


  
    return latlonglist


#this is how we evalute the fitness     
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0
    
    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance
    
    def routeFitness(self):
        if self.fitness == 0:
            # print (f"self.fitness  is 0\n self routdistance {self.routeDistance()}")
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness




def createRoute(latlonglist):
    '''
    
    this creats a random selection of potential permutations
    
    '''
    route = random.sample(latlonglist, len(latlonglist))
    return route





def initialPopulation(popSize, latlonglist):

    '''
    
    
    creates initial population of permuations. This is the base case from which the gentic alorithim starts. Afterwards it will swap using elites ranked by the fitness funtion
    These elites pass on their sequence to the next generation by mating. Then then there is some mutation that happens. All of that happens in later functions
    
    
    '''
    population = []

    for i in range(0, popSize):
        route=createRoute(latlonglist)
        
        population.append(route)
   

    # df=pd.DataFrame(dist)
    # print(f'this is the shape of ipop dist{df.shape}')
    return population


def gendf(pop,n):
    '''
    This function reads data, writes data to
    a csv file if a local file does not exist, and returns a df. It was created to gather information to do some stats testting
    '''
    if os.path.isfile(f'n_{n}pop.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv(f'n_{n}pop.csv',index_col=0)
       
        pop=pd.DataFrame(pop)
        if len(pop)==len(df):
            df.join(pop,how='outer')
            df.to_csv(f'n_{n}pop.csv')
        else:
            pop=pop.T
            df.join(pop,how='outer')
            df.to_csv(f'n_{n}pop.csv')

       
        
    else:
        
        df=pd.DataFrame(pop)
        df=df.T
        df.to_csv(f'n_{n}pop.csv')




def bigdf(popRanked,n):
    '''
    This function reads data, writes data to
    a csv file if a local file does not exist, and returns a df. It was created to gather information to do some stats testting
    '''
    if os.path.isfile(f'n_{n}big.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv(f'n_{n}big.csv',index_col=0)
       
        popRanked=pd.DataFrame(popRanked)
        if len(popRanked)==len(df):
            cols=["Index"]
            for i in range(1,(df.shape[1])):
             cols.append(f"gen{i+1}" )
            popRanked.columns=cols           
            df.merge(popRanked,how='outer')
            df.to_csv(f'n_{n}big.csv')
        else:
            for i in range(1,(df.shape[1])):
             cols.append(f"gen{i+1}" )
            popRanked.columns=cols  
            # df.merge(popRanked,how='cross',on=['Index'])
            df.merge(popRanked,how='outer')
            df.to_csv(f'n_{n}big.csv')

       
        
    else:
        
        df=pd.DataFrame(popRanked)
        df=df
        df.to_csv(f'n_{n}big.csv')
    

    
        
        
        
    
    
        

    

    
def rankRoutes(population):
    # gendf(population)
  
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

def selection(popRanked, eliteSize,n):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    bigdf(df,n)
    
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults
def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool
def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child
def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children







def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual







def mutatePopulation(population, mutationRate):
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop









def nextGeneration(currentGen, eliteSize, mutationRate,n):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize,n)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
   


  
   
    # df=pd.DataFrame(dist)
    # print(f'this is the shape of nextgen dist{df.shape}')
    return nextGeneration    







def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    n=len(population)
    print("Initial distance: " + f'{(1 / rankRoutes(pop)[0][1]):.2g}')
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate,n)
    
    print("Final distance: " + f'{(1 / rankRoutes(pop)[0][1]):.2g}')
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute


    




def plotyTSPoutput(compdf,show):





    optimal=compdf[['Near Optimal Ordered Lat', 'Near Optimal Ordered Long']]
    toappend=optimal.loc[0].to_dict()
    
    appendindex=len(optimal.index.to_list())
    # toappend
    toappend=pd.DataFrame(toappend,index=[appendindex])
    optimal=pd.concat([optimal,toappend],axis=0)
    
    
    optlat=optimal['Near Optimal Ordered Lat']
    optlong=optimal['Near Optimal Ordered Long']
    geo_df = gpd.GeoDataFrame(optimal, geometry=gpd.points_from_xy(optlong,optlat))
    geo_df=geo_df.rename(columns={'Near Optimal Ordered Lat':'lat', 'Near Optimal Ordered Long':'long'})
    names=[f'Point_{i}' for i in range(len(geo_df))]
    geo_df['names']=names
    geo_df= geo_df.set_crs(4326)
    codeupgeo=shapely.geometry.Point( -98.4895,29.42697)
    fig1 = go.Figure(go.Scattermapbox())
    for i in range(0,len(geo_df)-1):
        # x1=geo_df.geometry[i-2].x
        # y1=geo_df.geometry[i-2].y  
        x2=geo_df.geometry[i+1].x
        y2=geo_df.geometry[i+1].y
        x3=geo_df.geometry[i].x
        y3=geo_df.geometry[i].y   
        fig1.add_trace(go.Scattermapbox(
        mode = "markers+lines",
        lon = [x2,x3],
        lat = [y2,y3],
        marker = dict(size=7,opacity=.5,allowoverlap=True),
        line=dict(width=1),
        name=f'({y2},{x2})->'))
    
    
    
    fig1.add_trace(go.Scattermapbox(
        mode='markers',
        lon = [codeupgeo.x],
        lat = [codeupgeo.y],
        marker = dict(size=1500,color='green',opacity=.25,allowoverlap=True),
        showlegend=False))
    fig1.add_trace(go.Scattermapbox(
        mode='markers',
        lon = [codeupgeo.x],
        lat = [codeupgeo.y],
        marker = dict(size=10,color='red',opacity=1,allowoverlap=True),
        name='Code Up',
        )
       )
    fig1.update_layout(
    
        title=dict(text=f'Code Up TSP Simulation:\nn={len(geo_df)-1}',font=dict(color='white',family="Courier New"),x=0.98,y=0.95),
        height=900,
        margin ={'l':0,'t':0,'b':0,'r':0},
        mapbox = {
            'center': {'lon': codeupgeo.x, 'lat': codeupgeo.y},
            'style': "carto-darkmatter",
            # 'center': {'lon': -20, 'lat': -20},
            'zoom': 8.15},
        legend=dict(bgcolor='black',orientation='v',x=0.025,y=.5, borderwidth=1.5,font=dict(color='white',family="Courier New")))
    fig1.write_image(f'Code_Up_TSP_Simulation_n{len(geo_df)-1}.png',format='png',height=1500,width=2500,scale=1.25)
    # if show==True:
    #     fig1.show()
    
    
    
    
    
    
    if show==True:
        fig1.show()











def geneticAlgorithmProgressPlot(population,popSize, eliteSize, mutationRate, generations,show):
    fig = plt.figure(figsize=(15,15))
    n=len(population)
    ax = host_subplot(111, axes_class=AA.Axes)
    ax.autoscale()
    ax.margins(0.1)

    st = time.time()
   
    pop=initialPopulation(popSize, population)
    gendf(pop,n)

    popdf=[]
    popdf.append(pop)
    # df=pd.DataFrame(popdf)
    # print(f'this is the shape of 1st pop df after ip in genproplt{df.shape}')

    idist=("Initial distance: " + f'{(1 / rankRoutes(pop)[0][1]):.2g}\n')
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    
    for i in range(0, generations):
        pop= nextGeneration(pop, eliteSize, mutationRate,n)
        gendf(pop,n)
        popdf.append(pop)
        progress.append(1 / rankRoutes(pop)[0][1])
       
    fdist=("Final distance: " + f'{(1 / rankRoutes(pop)[0][1]):.2g}\n')
    # df=pd.DataFrame(popdf)
    # print(f'this is the shape of 1st pop dist inside genAlgPlot after adding gens dist{df.shape}')
     
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
   
 

    ax.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.title(f'Genetic Alg Performance for n={len(population)}')
    plt.savefig(fname=f'GeneticAlgoPlot_TSP_{len(population)}.png' ,dpi='figure',format='png')
   
    print(idist)
    print(fdist)
    elapsed_time = time.time() - st
    print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    if show == True:
        plt.show()
    
    # popdf=pd.DataFrame(popdf)
    # progressdf=pd.DataFrame(progress)

    return bestRoute



def geneticAlgorithm_with_Plot(n, popSize, eliteSize, mutationRate, generations,show=False):
    population=latlonglist(n,show)
   


    
    
   
    bestRoute=geneticAlgorithmProgressPlot(population,popSize, eliteSize, mutationRate, generations,show)
   
  
    if show== True:
        nearoptimalcircuit(bestRoute)
    formatedpop=[(float(f'{p.y:.4g}'),float(f'{p.x:.4g}')) for p in population]
    bestrouteformated=[(float(f'{p.y:.4g}'),float(f'{p.x:.4g}')) for p in bestRoute]
#    bestroute

    b=pd.DataFrame(bestrouteformated,columns=['Near Optimal Ordered Lat','Near Optimal Ordered Long'])
    a=pd.DataFrame(formatedpop,columns=['Random Order Lat','Random Order Long'])
    compdf=pd.concat([a,b],axis=1)
    print('\n\n\n')
   
    display(compdf)
    plotyTSPoutput(compdf,show)    
   


   


    return bestRoute






def strtuppletotuupple(col):
    for i in range(0,len(col)):
        a=(col[i]).replace('(','').replace(')','').split(',')
        a[0]=float(a[0])
        a[1]=float(a[1])
        col[i]=a
    return col

def strtuppletotuuppleouter(df):
    newdf=pd.DataFrame()
    cols=df.columns.to_list()

    for c in cols:
        col=df[c]
        col=strtuppletotuupple(col)
        newdf[f'{c}']=pd.Series(col)
    return newdf


def latlongtodis(df):
    cols=df.columns.to_list()
    bigdisdf=pd.DataFrame()
    for i in cols:
        disdf=pd.DataFrame()
        col=df[i]
        distcol=[]
        for c in range(1,len(col)):
            x1=col[c-1][0]
            x2=col[c][0]
            y1=col[c-1][1]
            y2=col[c][1]
            dist=((x1-x2)**2+(y1-y2)**2)**(.5)
            distcol.append(dist)
        dista=pd.DataFrame(distcol)
        disdf=pd.concat([dista,disdf],axis=1)
        bigdisdf=pd.concat([disdf,bigdisdf],axis=1)
    return bigdisdf


def latlongstringtouppletoeucliddist(df):
    
    da=strtuppletotuuppleouter(df)
    db=latlongtodis(da)
    return db



    
def bestRoutedist(bestRoute,n):
   
   
    df=pd.DataFrame(bestRoute)
    df.to_csv(f'n_{n}bestroute.csv')

    return 
    




def popandbestcsv(n):
    df = pd.read_csv(f'n_{n}pop.csv', index_col=0)
    bestroute = pd.read_csv(f'n_{n}bestroute.csv', index_col=0)
    # df=pd.concat( [df ,bestroute],axis=1)
    # intfor=int(len(df))
    # cols=[i for i in range(len(df.T))]
    # df = pd.DataFrame('x', index=range(intfor), columns=cols)
    


    dist=latlongstringtouppletoeucliddist(df)


    total_mean=dist.mean(axis=0)
    total_var=dist.var(axis=0)
    total_skew=dist.skew(axis=0)
    total_kurtosis=dist.kurtosis(axis=0)
    total_sum=dist.sum(axis=0)



    statsuff=[total_mean,total_var,total_skew,total_kurtosis,total_sum]
   
    df=pd.concat(statsuff,axis=1)


    columns=['mean','var','kurt','skew','sum']
    colnums=list(range(len(columns)))
    colmap=dict(zip(colnums,columns))
    statdf=df.rename(columns=colmap)
    cols=[]
    for i in range(0,(dist.shape[1])):
        cols.append(f"gen{i+1}" )
    dist.columns=cols



    comparesum=statdf['sum'].aggregate(['min','max','mean'])
    return statdf,comparesum,bestroute,dist
    

















def bestdist(best):

    best=best['0']
    xs=[]
    ys=[]

    best=[(i.replace('(','').replace(')','').replace("'",'').split(',')) for i in best]
    b=best
    bestdist=[]
    for i in range(1,len(best)):
        x1=b[i][0]
        x2=b[i-1][0]
        y1=b[i][1]
        y2=b[i-1][1]
        x1=float(x1)
        x2=float(x2)
        y1=float(y1)
        y2=float(y2)

        dist=(((x1-x2)**2)+((y1-y2)**2))**.5
        bestdist.append(dist)


    
    euclidieanbesttotal=sum(bestdist)
    return euclidieanbesttotal



def focusedforreport(n):
    stats,comp,bestroute,dist=popandbestcsv(n)
    TSPFocusedViz(n)
    # besd=latlongstringtouppletoeucliddist(bestroute)
    besd=bestdist(bestroute)
    comp=pd.DataFrame(comp)
    comp['diff%']=((comp['sum']-besd)/comp['sum'])*100
   
    return stats, comp,besd,dist




def looptoseeallbestvsminmaxmean():
    listtoindex=list(range(3,12,1))
    listtoadd=list(range(12,66,6))
    listtoindex.extend(listtoadd)
    compbigdf=pd.DataFrame()
   
    for i in listtoindex:
        n=i
        stats,comp,bestroute,dist=popandbestcsv(n)
        # TSPFocusedViz(n)
        # besd=latlongstringtouppletoeucliddist(bestroute)
        besd=bestdist(bestroute)
        comp=pd.DataFrame(comp)
        comp['diff%']=((comp['sum']-besd)/comp['sum'])*100
        compbigdf=pd.merge(left=compbigdf,right=comp,how='left',left_index=True,right_index=True,suffixes=[f'{n}',f'{n+1}'])


   
    return compbigdf


##stats
def genmeancomp(dist,a,b,alpha=.05):
    '''
    wecompare the distance generator by 100 different populations
    
    '''
    colset=set(dist.columns)
    twocombos=list(combinations(colset,2))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    fig.suptitle('Comp')

   
    
   

    


    cola=dist[a]
    colb=dist[b]
    sns.histplot(ax=axes[0],data=cola,kde=True)
    axes[0].set_title(a)
    sns.histplot(ax=axes[1],data=colb,kde=True)
    axes[1].set_title(b)
    plt.show()
    # g.add_legend(f'We compare:\n{str(i[0])} vs {str(i[1])}')
  
    t,p =stats.levene(cola,colb)
    if p < alpha:
        varequal=False
    else:
        varequal=True
    


    
    t,p = stats.ttest_ind(cola, colb,equal_var=varequal)
    nullsym=symbols('H_{0}')
    rejnull=symbols('Reject~H_{0}~?')
    null='The null hypothesis is that our populations are statistically the same.'

    if p / 2 > alpha:
        
        equalpopstring=f'No, we observe that {a} and {b} are statistically the same:'
        display(nullsym,null,rejnull)
        print(f"{equalpopstring}\nHence, we fail to reject our null hypothesis\n\n")
    elif t < 0:
    
        equalpopstring=f'No, we observe that {a} and {b} are statistically the same:'
        display(nullsym,null,rejnull)
        print(f"{equalpopstring}\nHence, we fail to reject our null hypothesis\n\n")
    else:
        
        equalpopstring=f'Yes, we observe that {a} and {b} are statistically different'
        display(nullsym,null,rejnull)
        print(f"{equalpopstring}\nHence, we reject our null hypothesis\n\n")  





def datadict(df):
    x=(pd.concat([df.dtypes,df.nunique(),df.count(),df.isnull().sum(),df[df.isnull()==0].kurtosis()],axis=1))
    type(x)
    x=x.reset_index()
    collist=x.columns.to_list()
    columns=['name','data type','unique','total count','null count','non null kurt']
    coldict=dict(zip(collist,columns))
    x.rename(columns=coldict,inplace=True)
    x.sort_values(by=['unique','total count','name'],inplace=True)
    x=x.reset_index(drop=True)
    x['percent null']=(x['null count']/ x['total count'])*100
    
   
    return x