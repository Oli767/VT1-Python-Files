# Import of Packages
import numpy as np
import pandas as pd
import statistics as st
import time

# Import of Packages for Functions
import math
import matplotlib.pyplot as plt
import sys

# Importing the Functions File
import Functions_VT2 as fn

# Importing Packets for the Genetic Algorithm
import random
from deap import base, creator, tools


def generate_scenarios(mu, sigma, Dt0, dt, Fth=20, Forecasts=100):
    """This function is calculating a denfined numer of Scenarios

    Args:
        mu          Mean Percentage Growth mu                       float
        sigma       Standart Deviation of Percentage Growth         float
        Dt0         Initial Demand t0                               int
        dt          Duration of Delta t in Years                    int
        Fth         Forecast time horizon                           int
        Forecasts   # number of Forecasts (+1 case we start at 0)   int
    Returns:
        Szenarios                                                   np.array

    To Call the Function use following syntax:
        Scenario_creation(mu, sigma, Dt0, dt, Fth, Forecasts)
    """
    np.random.seed(1)

    # Create arrays for indices
    S = np.arange(0, Forecasts)
    S2 = np.arange(0, Fth + 1)

    # Random values for spread of the scenario
    random_values = np.random.normal(0, 1, size=(len(S), len(S2) - 1))

    # Demand
    D = Dt0 * np.exp((mu * dt + sigma * np.sqrt(dt) * random_values).cumsum(axis=1))

    return D


def Scenario_plot(
    Scenarios,
    Fth=20,
    NoStep=True,
    Title="Demand Szenarios",
    label="Passenger Numbers",
    n=40,
):
    """This function is plotting the Scenarios Created in the Scenario function

    Args:
        Scenarios   Szenario Data                       np.array
        Fth         Forecast time horizon               int
        NoStep      Question if Step Plot or not        bool
        Title       Title for the Plot                  str
        ylabel      Y-Axis Description                  str
        n           Number of random selection          int
    Returns:
        Plot of all Demand Vectors in a Single Graph

    To Call the Function use following syntax:
        Scenario_plot(Scenarios, Fth)
    """
    # Adding one to the time horizon as Python starts counting from zero
    Fth += 1

    # If loop when Scenarios is only a Vector
    if Scenarios.ndim == 1:
        Scenarios = Scenarios.reshape(1, -1)
    else:
        Scenarios = Scenarios

    indices = np.random.choice(Scenarios.shape[0], size=(n))
    Small_Scenario = Scenarios[indices]

    plotvector = list(range(1, (Fth)))
    for scenario in Small_Scenario:
        if NoStep == True:
            plt.plot(plotvector, scenario, label="Scenario")
        else:
            plt.step(plotvector, scenario, where="post", label="Scenario")
    plt.grid(True)
    plt.xlabel("Years")
    plt.ylabel(label)
    plt.title(Title)
    plt.figure()


def Capacity(K0, delta_K, Forecasts):
    """
    This Function returns the Capacity value in Matrix format for a given Initial
    Capacity and delta Capacity Vector, copied according to the number of Forecasts

    Args:
        K0 (int): Initial Capacity
        delta_K (ndarray): delta Capacity Vector
        Forecasts (int): Number of forecasts

    Returns:
        K (ndarray): Capacity Matrix
    """
    # Repeat the delta_K vector 'Forecasts' times
    repeated_delta_K = np.repeat(delta_K[np.newaxis, :], Forecasts, axis=0)

    # Create a cumulative sum array starting from K0 for each forecast
    K = K0 + np.cumsum(repeated_delta_K, axis=1) * 1000000

    return K


def Revenue(K, D, r_K, r_K_rent, r_D):
    """
    This Function calculates the Revenue

    Args:
        K (ndarray): Capacity Vector
        D (ndarray): Demand Matrix
        r_K (float): _description_
        r_K_rent (float): _description_
        r_D (float): _description_

    Returns:
        Total_Revenue (float): Revenue
    """
    diff = K - D
    greater_zero = np.greater(diff, 0).astype(int)
    less_equal_zero = np.less_equal(diff, 0).astype(int)
    # if Overcapacity only amount of Demand can be sold
    rev_overcapacity = greater_zero * (D * r_K + D * r_K_rent + D * r_D)
    # if Undercapacity only available Capacity can be sold
    rev_undercapacity = less_equal_zero * (K * r_K + K * r_K_rent + K * r_D)
    Total_Revenue = rev_overcapacity + rev_undercapacity

    return Total_Revenue


def Cost(K, D, delta_K, co_K, co_D, ci_K, EoS, h):
    """
    This Function calculates the Revenue

    Args:
        K (ndarray): Capacity Vector
        D (ndarray): Demand Matrix
        delta_K (ndarray): delta Capacity Vector
        co_K (float): Operating Cost per Unit of Capacity
        co_D (float): Operating Cost per Unit of Demand
        ci_K (float): Installation Cost per Unit of Capaciy
        EoS (float): Economy of Scale Factor
        h (int): _description_

    Returns:
        Total_Cost (float): Cost
    """
    diff = K - D
    # Penalty Cost Overcapacity
    pc_over = 1
    # Penalty Cost Undercapacity
    pc_under = 1
    # Create an Index Matrix with the Condition for undercapacity
    cos_overcapacity = np.greater(diff, 0).astype(int)
    cos_undercapacity = np.less(diff, 0).astype(int)
    cos_equalcapacity = np.equal(diff, 0).astype(int)

    Total_Cost = (
        ((ci_K * (delta_K) ** EoS) / h)
        + cos_undercapacity * (pc_under + (co_D * K + co_K * K))
        + cos_overcapacity * (pc_over + (co_D * D + co_K * D))
        + cos_equalcapacity * (co_D * D + co_K * K)
    )
    return Total_Cost


def NPV_calculation(K, D, delta_K, Param):
    """
    This Function calculates the Net Present Value by calling the Revenue and Cost
    Functions

    Args:
        Revenue (ndarray): Total Revenue
        Cost (ndarray): Total Cost
        discount (float): discount rate

    Returns:
        NPV (ndarray): Net Present Value
    """
    # Parameters
    r_D = Param["r_D"]  # Revenues per Unit of Demand per Period
    r_K = Param["r_K"]  # Revenues per Unit of Capacity per Period
    r_K_rent = Param["r_K_rent"]  # Rental Revenues per Unit of Capacity per Period
    co_K = Param["co_K"]  # Operational costs per unit of capacity per period
    co_D = Param["co_D"]  # Operational cost per unit of demand per period
    ci_K = Param["ci_K"]  # Installation cost per unit of capacity
    discount = Param["discount"]  # Discount factor
    EoS = Param["EoS"]  # EoS factor
    h = Param["h"]
    Fth = Param["Fth"]
    dt = Param["dt"]

    # Revenue
    Rev = Revenue(K, D, r_K, r_K_rent, r_D)

    # Cost
    Cos = Cost(K, D, delta_K, co_K, co_D, ci_K, EoS, h)

    # Plus one because Python starts at Zero
    t = 1 + np.arange(0, Fth, dt)
    Profit = Rev - Cos
    Discount = 1 / (1 + discount) ** t
    Present_value = Profit * Discount
    NPV = np.sum(Present_value, axis=1)

    return NPV


def ENPV_calculation(delta_K, Param, D):
    """
    This Function calculates the Expected Net Present Value by Calling the NPV
    Calculation Function

    Args:
        delta_K (ndarray): Capacity increase vector
        Param (dictionary): Parameter Dictionary

    Returns:
        ENPV (ndarray): Expected Net Present Value
    """
    K0 = Param["K0"]
    Forecasts = Param["Forecasts"]
    K = Capacity(K0, delta_K, Forecasts)
    ENPV = np.mean(NPV_calculation(K, D, delta_K, Param))

    return ENPV


def GA(Param, D):
    """
    This is a Genetic Algorithm seeking to find an optimal delta_K Vector to
    maximise the ENPV

    Args:
        Param (Dictionary): Parameter Dictionary

    Returns:
        delta_K (ndarray): _description_
    """

    # Define the vector of values
    value_vector = Param["allowed_values"]

    # Create the DEAP framework
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    # Define the individual initialization function
    def init_individual():
        return np.array([random.choice(value_vector) for _ in range(Param["Fth"])])

    # Register the initialization function and the population function
    toolbox.register(
        "individual", tools.initIterate, creator.Individual, init_individual
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Mutation operator
    def mutate_individual(individual):
        for i in range(len(individual)):
            if random.random() < 0.2:  # Mutation probability
                individual[i] = random.choice(value_vector)

    # Register the mutation operator
    toolbox.register("mutate", mutate_individual)

    # Define the evaluation function
    def evaluate(individual, Param=Param):
        return (ENPV_calculation(individual, Param=Param, D=D),)

    toolbox.register("evaluate", evaluate)

    # Define the genetic operators
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Set up the genetic algorithm
    population = toolbox.population(n=Param["population"])
    cxpb, mutpb, ngen = 0.5, 0.2, 10

    # Perform the evolution
    for gen in range(ngen):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        population[:] = offspring

    # Get the best individual
    best_ind = tools.selBest(population, 1)[0]

    return best_ind


def CDF_Plot(
    Vector1,
    Vector2,
    label1="Example CDF Curve",
    label2="Optimized CDF Curve",
    label3="Example ENPV",
    label4="Optimized ENPV",
):
    """This function is Plotting the Cumulative Density Function of the NPVs
    Args:
        Vector1         Traditional Input Vector 1              np.array
        Vector1         Flexible Input Vector 1                 np.array

    Returns:
        Plot of all input Vectors in a CDF Graphic
        Percentiles     10th, 90th Percentile Input Vectors     np.array

    To call this Function use following syntax:
        CDF_Plot(Vector1, Vector2)
    """
    percentile_10a = np.percentile(Vector1, 10)
    percentile_90a = np.percentile(Vector1, 90)
    percentile_10b = np.percentile(Vector2, 10)
    percentile_90b = np.percentile(Vector2, 90)

    # Creating a subplot
    fig, ax = plt.subplots()

    # Step plot code with specific values
    ax.plot(
        np.sort(Vector1),
        np.arange(1, len(Vector1) + 1) / float(len(Vector1)),
        linestyle="-",
        label=label1,
        linewidth=2,
        color="green",
        alpha=0.7,
    )

    ax.plot(
        np.sort(Vector2),
        np.arange(1, len(Vector2) + 1) / float(len(Vector2)),
        linestyle="-",
        label=label2,
        linewidth=2,
        color="blue",
        alpha=0.7,
    )

    mean1 = np.mean(Vector1)
    Vector3 = np.full_like(Vector1, mean1)
    ax.plot(
        np.sort(Vector3),
        np.arange(1, len(Vector3) + 1) / float(len(Vector3)),
        linestyle="--",
        label=label3,
        linewidth=2,
        color="green",
        alpha=0.7,
    )
    mean2 = np.mean(Vector2)
    Vector4 = np.full_like(Vector2, mean2)
    ax.plot(
        np.sort(Vector4),
        np.arange(1, len(Vector4) + 1) / float(len(Vector4)),
        linestyle="-.",
        label=label4,
        linewidth=2,
        color="blue",
        alpha=0.7,
    )
    ax.axhline(
        0.9,
        color="orange",
        linestyle="--",
        label="90th Percentile",
    )

    ax.axhline(
        0.1,
        color="red",
        linestyle="-.",
        label="10th Percentile",
    )

    # Add crosshair at the specified point
    ax.plot(percentile_90a, 0.9, marker="X", color="black", markersize=6)
    ax.plot(percentile_10a, 0.1, marker="X", color="black", markersize=6)
    ax.plot(percentile_90b, 0.9, marker="X", color="black", markersize=6)
    ax.plot(percentile_10b, 0.1, marker="X", color="black", markersize=6)

    ax.grid(True)
    ax.set_title("Cumulative Distribution Function (CDF)")
    ax.set_xlabel("NPVs")
    ax.set_ylabel("Cumulative Probability [%]")
    ax.legend()
    plt.show()
    percentiles = [percentile_10a, percentile_90a, percentile_10b, percentile_90b]
    return percentiles
