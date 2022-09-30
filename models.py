
from __future__ import division

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


from itertools import permutations

from matplotlib_scalebar.scalebar import ScaleBar

import time
import seaborn as sns



import plotly.express as px
import geopandas as gpd
import shapely
import numpy as np
import wget

from math import radians, cos, sin, asin, sqrt, factorial



from sympy import symbols, sympify
from sympy.plotting import plot

from sympy import S, I, pi, gamma
from sympy.abc import x






def showfactorials():
    n=list(range(0,125,5))
   
    x = symbols('x',integer=True)

    plot(gamma(x), show=True,xlim=(0,15))
   
    sleep(15)
    clear_output()
    print('Now we show 0 to 125 at intervals of 5:\n')
    sleep(5)
    clear_output()



    for i in n:
        h=f'{factorial(i)}'
        print('\n')
        for char in h:
            sleep(0.0009)
            print(char, end='', flush=True)
            
   
   
   























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


    rangeforloop=list(range(1,len(bestroute)))
    if len(bestroute)%2!=0:
        for i in rangeforloop:
            subline=[]
            if i==1:
                x1=bestroute[i-1].x
                y1=bestroute[i-1].y
                x2=bestroute[i].x
                y2=bestroute[i].y
                ax.plot(x1,y1,color="green", marker="o")
                ax.plot([x1,x2],[y1,y2],color="green",   linestyle=":")
            elif i==rangeforloop[-1]:
                x1=bestroute[i-1].x
                y1=bestroute[i-1].y
                x2=bestroute[i].x
                y2=bestroute[i].y
                x3=bestroute[0].x
                y3=bestroute[0].y

                ax.plot([x1,x2],[y1,y2],color="white",marker='o' ,   linestyle=":")
                ax.plot([x2,x3],[y2,y3],color="red",   linestyle=":")
                ax.plot(x1,y1,color="blue")
                ax.plot(x3,y3,color="red", marker="x")

            elif i%2==0:               
                x1=bestroute[i-1].x
                y1=bestroute[i-1].y
                x2=bestroute[i].x
                y2=bestroute[i].y
                ax.plot([x1,x2],[y1,y2],color="white",marker='o'  ,linestyle=":")
                # x.append(x1)
                # y.append(y1)
                # path.append(i)
            else:
                x1=bestroute[i-1].x
                y1=bestroute[i-1].y
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
                x1=bestroute[i-1].x
                y1=bestroute[i-1].y
                x2=bestroute[i].x
                y2=bestroute[i].y
                ax.plot(x1,y1,color="green", marker="o")
                ax.plot([x1,x2],[y1,y2],color="green",   linestyle=":")
            elif i==rangeforloop[-1]:
                x1=bestroute[i-1].x
                y1=bestroute[i-1].y
                x2=bestroute[i].x
                y2=bestroute[i].y
                x3=bestroute[0].x
                y3=bestroute[0].y

                ax.plot([x1,x2],[y1,y2],color="white",marker='o' ,   linestyle=":")
                ax.plot([x2,x3],[y2,y3],color="red",   linestyle=":")
                ax.plot(x1,y1,color="blue")
                ax.plot(x3,y3,color="red", marker="x")

            elif i%2!=0:               
                x1=bestroute[i-1].x
                y1=bestroute[i-1].y
                x2=bestroute[i].x
                y2=bestroute[i].y
                ax.plot([x1,x2],[y1,y2],color="white",marker='o'  ,linestyle=":")
                # x.append(x1)
                # y.append(y1)
                # path.append(i)
            else:
                x1=bestroute[i-1].x
                y1=bestroute[i-1].y
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

    

# Each are working I will try to combine them
def create_random_point(x0,y0,distance):
    """
            Utility method for simulation of the points
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
    route = random.sample(latlonglist, len(latlonglist))
    return route





def initialPopulation(popSize, latlonglist):
    population = []

    for i in range(0, popSize):
        route=createRoute(latlonglist)
        
        population.append(route)
    dist=[]

   
    for i in range(1,len(population)-1):
        x1=population[i][0].x
        x2=population[i-1][0].x
        y1=population[i][1].y
        y2=population[i-1][1].y
        dist.append((((x1-x2)**2)+((x1-x2)**2))**.5)
    return population,dist




    
def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
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


def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    dist=[]

    for a in range(0,len(nextGeneration)):
        for b in range(0,len(nextGeneration[a])):
            for i in range(1,len(nextGeneration)-1):
                x1=nextGeneration[i][0].x
                x2=nextGeneration[i-1][0].x
                y1=nextGeneration[i][1].y
                y2=nextGeneration[i-1][1].y
                dist.append((((x1-x2)**2)+((x1-x2)**2))**.5)
    return nextGeneration,dist    
def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + f'{(1 / rankRoutes(pop)[0][1]):.2g}')
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
    
    print("Final distance: " + f'{(1 / rankRoutes(pop)[0][1]):.2g}')
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute


    

    
# def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
#     pop = initialPopulation(popSize, population)
#     progress = []
#     progress.append(1 / rankRoutes(pop)[0][1])
    
#     for i in range(0, generations):
#         pop = nextGeneration(pop, eliteSize, mutationRate)
#         progress.append(1 / rankRoutes(pop)[0][1])
    
#     plt.plot(progress)
#     plt.ylabel('Distance')
#     plt.xlabel('Generation')
#     plt.savefig(fname=f'GeneticAlgoPlot_TSP_{len(population)}.png' ,dpi='figure',format='png')
  
#     plt.show()




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
    for i in range(1,len(geo_df)):
        # x1=geo_df.geometry[i-2].x
        # y1=geo_df.geometry[i-2].y  
        x2=geo_df.geometry[i-1].x
        y2=geo_df.geometry[i-1].y
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
    ax = host_subplot(111, axes_class=AA.Axes)
    ax.autoscale()
    ax.margins(0.1)

    st = time.time()
    dist=[]
    pop,d1 = initialPopulation(popSize, population)
    dist.append(d1)
    popdf=[]
    popdf.append(pop)

    idist=("Initial distance: " + f'{(1 / rankRoutes(pop)[0][1]):.2g}\n')
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    
    for i in range(0, generations):
        pop,d2 = nextGeneration(pop, eliteSize, mutationRate)
        pd.DataFrame(pop)
        popdf.append(pop)
        progress.append(1 / rankRoutes(pop)[0][1])
        dist.append(d2)
    
    fdist=("Final distance: " + f'{(1 / rankRoutes(pop)[0][1]):.2g}\n')
     
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

    return bestRoute,dist



def geneticAlgorithm_with_Plot(n, popSize, eliteSize, mutationRate, generations,show=False):
    population=latlonglist(n,show)
   


    
    
   
    bestRoute,dist=geneticAlgorithmProgressPlot(population,popSize, eliteSize, mutationRate, generations,show)
   
  
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
   


   


    return bestRoute,dist









    
