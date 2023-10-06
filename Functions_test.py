# Import of Packages
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


def Scenario(mu, sigma, Dt0, t=1, Fth=50, Forecasts=20):
    """This function is calculating various scenarios for Demand
    Args:
        mu          Mean Percentage Growth mu
        sigma       Standart Deviation of Percentage Growth
        Dt0          Initial Demand t0
        Delta t     Duration of Delta t in Years
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
                D[j - 1] * mu * t
                + D[j - 1] * sigma * random.choice(randomrange) * math.sqrt(t)
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

# Scenario_plot(Scenario(mu, sigma, Dt0, t, Fth, Forecasts), Fth)
