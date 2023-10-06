# Import of Packages
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


def Scenario_creation(mu, sigma, Dt0, dt=1, Fth=50, Forecasts=20):
    """This function is calculating various scenarios for Demand
    Args:
        mu          Mean Percentage Growth mu
        sigma       Standart Deviation of Percentage Growth
        Dt0         Initial Demand t0
        dt          Duration of Delta t in Years
        Fth         Forecast time horizon
        #           number of Forecasts (+1)
    Returns:
        Szenarios
    """
    S = list(range(1, Forecasts + 1))
    S2 = list(range(1, Fth))

    # Spread of the Scenario can be adjusted here at randomrange
    randomrange = np.arange(-1, 1, 0.1)
    Szenarios = []

    for i in S:
        D = [Dt0]
        for j in S2:
            Szenario = D[j - 1] + (
                D[j - 1] * mu * dt
                + D[j - 1] * sigma * random.choice(randomrange) * math.sqrt(dt)
            )
            D.append(Szenario)
        Szenarios.append(D)
    return Szenarios


def Scenario_plot(Scenarios, Fth=50):
    """This function is plotting the Scenarios Created in the Scenario function
    Args:
        Szenario Data
        Fth         Forecast time horizon
    Returns:
        Plot of Vectors
    """
    plotvector = list(range(1, (Fth + 1)))
    for scenario in Scenarios:
        plt.plot(plotvector, scenario, label="Scenario")
    plt.grid(True)
    plt.xlabel("Years")
    plt.ylabel("Passenger Numbers")
    plt.title("Demand Szenarios for Zurich Airport")
    plt.figure()


# Required Parameters
# mu = 0.042754330256447565
# sigma = 0.05891802084811409
# Dt0 = 22561132
# t = 1
# Fth = 50
# Forecasts = 20

# Scenario_plot(Scenario(mu, sigma, Dt0, dt, Fth, Forecasts), Fth)


def NPV_Calculation(D, K, t, r_D, r_K, co_K, co_D, ci_K, discount, EoS):
    """This function is calculates the NPV as a function of a Demand and Capacity Vector
    Args:
        r_D         Revenues per Unit of Demand per Period
        r_K         Revenues per Unit of Capacity per Period
        co_K        Operational costs per unit of capacity per period
        co_D        Operational cost per unit of demand per period
        ci_K        Installation cost per unit of capacity
        discount    Discount factor
        EoS         Economy of Scale factor
    Returns:
        NPV for given Inputs
    """
    deltaK = []
    for i in range(len(K)):
        if i == 0:
            # For the first element, just append it as is.
            deltaK.append(0)
        else:
            # For subsequent elements, subtract the previous element.
            difference = K[i] - K[i - 1]
            deltaK.append(difference)

    Revenue = []
    Cost = []
    Profit = []
    Discount = []
    for i in range(len(K)):
        # Revenue
        R = r_D * D[i] + r_K * K[i]
        Revenue.append(R)
        # Cost
        # Investment costs
        CI = ci_K * deltaK[i] ** EoS
        # Operational costs
        CO = co_K * K[i] + co_D * D[i]
        C = CI + CO
        Cost.append(C)
        # Profits
        P = R - C
        Profit.append(R)
        # Discount
        PV = (1 / (1 + discount) ** t) * P
        Discount.append(PV)
    NPVs = sum(Discount)
    NPV = sum(NPVs)
    return NPV
