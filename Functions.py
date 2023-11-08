# Import of Packages
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


def Scenario_creation(mu, sigma, Dt0, dt=1, Fth=50, Forecasts=20):
    """This function is calculating various scenarios for Demand

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
    Fth += 1
    # Creation of Number of Forecast Vector
    S = list(range(1, Forecasts + 1))
    # Creation of a time length of the Scenario Vectors
    S2 = list(range(1, Fth))

    # Spread of the Scenario can be adjusted here at randomrange
    randomrange = np.arange(-1, 1, 0.1)
    # Initialise a Scenarios Vector
    Szenarios = []
    # For loop to iterate over all Forecasts
    for i in S:
        # Add Demand at t0 to the Vector
        D = [Dt0]
        # Second for loop to iterate over the time length of the Scenarios
        for j in S2:
            Szenario = D[j - 1] + (
                D[j - 1] * mu * dt
                + D[j - 1] * sigma * random.choice(randomrange) * math.sqrt(dt)
            )
            # Append all individual Demand Curves to the Scenarios
            D.append(Szenario)
        # Append all Demand Vectors to the Scenarios Maxtrix
        Szenarios.append(D)
    # Change Shape to an Numpy Array
    Szenarios = np.array(Szenarios)
    # Get rid of the Initial Demand Value
    Szenarios = Szenarios[:, 1:]
    return Szenarios


def Scenario_plot(Scenarios, Fth=50, Title="Demand Szenarios"):
    """This function is plotting the Scenarios Created in the Scenario function

    Args:
        Scenarios   Szenario Data                  np.array
        Fth         Forecast time horizon          int
        Title       Title for the Plot             str
    Returns:
        Plot of all Demand Vectors in a Single Graph

    To Call the Function use following syntax:
        Scenario_plot(Scenarios, Fth)
    """
    # If loop when Scenarios is only a Vector
    if Scenarios.ndim == 1:
        Scenarios = Scenarios.reshape(1, -1)
    else:
        Scenarios = Scenarios

    plotvector = list(range(1, (Fth + 1)))
    for scenario in Scenarios:
        plt.plot(plotvector, scenario, label="Scenario")
    plt.grid(True)
    plt.xlabel("Years")
    plt.ylabel("Passenger Numbers")
    plt.title(Title)
    plt.figure()


# Required Parameters
# mu = 0.042754330256447565
# sigma = 0.05891802084811409
# Dt0 = 22561132
# t = 1
# Fth = 50
# Forecasts = 20
# To Call the Function use this Format:
# Scenario_plot(Scenario_creation(mu, sigma, Dt0, dt, Fth, Forecasts), Fth)


def NPV_Calculation(
    D,
    K,
    Fth,
    dt,
    th=1000000,
    r_D=0.03,
    r_K=0.03,
    r_K_rent=0.03,
    co_K=0.01,
    co_D=0.004,
    ci_K=10,
    discount=0.05,
    EoS=0.85,
):
    """This function is calculates the NPV as a function of a Demand and Capacity Vector

    Args:
        D           Demand Vector                                       np.array
        K           Estimated Capacity Vector                           np.array
        Fth         Forecast time horizon                               int
        dt          Steptime [Years]                                    int
        th          Throughput Capacity per Unit of Capacity            int
        r_D         Revenues per Unit of Demand per Period              float
        r_K         Revenues per Unit of Capacity per Period            float
        r_K_rent    Rental Revenues per Unit of Capacity per Period     float
        co_K        Operational costs per unit of capacity per period   flaot
        co_D        Operational cost per unit of demand per period      float
        ci_K        Installation cost per unit of capacity              float
        discount    Discount factor                                     float
        EoS         Economy of Scale factor                             float

    Returns:
        NPV for given Inputs                                            np.array

    To call this Function use following syntax:
        NPV_Calculation(D, K, Fth, dt, r_D, r_K, co_K, co_D, ci_K, discount, EoS)
    """
    Fth += 1
    # Creation of a time vector
    t = np.arange(1, Fth, dt)

    # Creation of a Capacity Change Vector
    deltaK0 = np.diff(K)
    # Setting the initial Value of the Change Vector to Zero
    deltaK = np.insert(deltaK0, 0, 0)

    # If loop when the Demand Matrix is only a Vector
    if D.ndim == 1:
        D = D.reshape(1, -1)
    else:
        D = D
    # Calculate the Difference Matrix
    diff = K - D
    # Initialize an array to store Revenue for each value
    Revenue = np.zeros(D.shape)
    # Initialize an array to store NPV for each row
    NPV = np.zeros(D.shape[0])
    # For loop to iterate over all Scenarios
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            # If condition to account for over- or undercapacity
            if diff[i, j] > 0:
                Revenue[i, j] = (
                    D[i, j] * r_K_rent + K[j] * r_K + th * D[i, j] * r_D
                ) - ((K[j] - D[i, j]) * r_K_rent + (K[j] - D[i, j]) * th * r_K)
            else:
                Revenue[i, j] = K[j] * r_K_rent + K[j] * r_K + th * D[i, j] * r_D

        # Operationals Costs as a function of Demand and Capacity
        Cost_Ops = co_K * K + co_D * th * D
        # Investment Costs as a function of Capacity and Economys of Scale Factor
        Cost_Investment = ci_K * (np.power(deltaK, EoS))
        # Total Cost as Sum of Operational and Investment Costs
        Cost = Cost_Ops + Cost_Investment
        # Profit as Difference of Revenue and Total Cost
        Profit = Revenue - Cost
        # Discount rate as a function of the Discount Factor and time t
        Discount = 1 / (1 + discount) ** t
        # Present Value as a function of Discount rate and Profit
        Present_Value = Discount * Profit
        # Net Present Value as sum of all Present Values
        NPV[i] = np.sum(Present_Value[i])
    return NPV


def Decision_Rule(D, K0, deltaK_Flex):
    """This function is creating new Capacity Vectors while considering a decision rule

    Args:
        D               Demand Vector                       np.array
        K0              Initial Capacity                    integer
        deltaK_flex     Capacity increase vector            list with 3 values

    Returns:
        K_Flex          Capacity vector considering a decision rule    np.array

    To call this Function use following syntax:
        Decision_Rule(D, K0, deltaK_Flex)
    """
    # If loop when the Demand Matrix is only a Vector
    if D.ndim == 1:
        D = D.reshape(1, -1)
    else:
        D = D
    # Create an array of the same shape as D initialized with K0
    K_Flex = np.full(D.shape, K0, dtype=D.dtype)
    # For loop to iterate over all Scenarios
    for i in range(D.shape[0]):
        # For loop to iterate over all values of a Scenario
        for j in range(D.shape[1]):
            # if condition to check for overcapacity
            if (K_Flex[i, j - 1] - D[i, j]) >= 0:
                new_capacity = K_Flex[i, j - 1]
            # elif condition to check the severity of the capacity deficit
            elif (K_Flex[i, j - 1] - D[i, j]) < -deltaK_Flex[0]:
                new_capacity = K_Flex[i, j - 1] + deltaK_Flex[1]
            # elif condition to check the severity of capacity deficit
            elif (K_Flex[i, j - 1] - D[i, j]) < -deltaK_Flex[1]:
                new_capacity = K_Flex[i, j - 1] + deltaK_Flex[2]
            # else condition to check the severity of the Capacity deficit
            else:
                new_capacity = K_Flex[i, j - 1] + deltaK_Flex[0]
            # changing the Capacity values for the given overcapacity or deficit
            K_Flex[i, j] = new_capacity
    return K_Flex


def Decision_Rule_Excel(D, K0=25, deltaK_Flex=5):
    """This function is creating new Capacity Vectors while considering a decision rule

    Args:
        D               Demand Vector                       np.array
        K0              Initial Capacity                    integer
        deltaK_flex     Capacity increase vector            int

    Returns:
        K_Flex          Capacity vector considering a decision rule    np.array

    To call this Function use following syntax:
        Decision_Rule(D, K0, deltaK_Flex)
    """
    # If loop when the Demand Matrix is only a Vector
    if D.ndim == 1:
        D = D.reshape(1, -1)
    else:
        D = D
    # Create an array of the same shape as D initialized with K0
    K_Flex = np.full(D.shape, K0, dtype=D.dtype)
    # For loop to iterate over all Scenarios
    for i in range(D.shape[0]):
        # For loop to iterate over all values of a Scenario
        for j in range(D.shape[1]):
            # if condition to check for overcapacity
            if (K_Flex[i, j - 1] - D[i, j]) >= 0:
                new_capacity = K_Flex[i, j - 1]
            # else condition to check  for undercapacity
            else:
                new_capacity = K_Flex[i, j - 1] + deltaK_Flex
            # changing the Capacity values for the given overcapacity or deficit
            K_Flex[i, j] = new_capacity
    return K_Flex


def Flex_NPV_Calculation(
    D,
    K,
    Fth,
    dt,
    th=1000000,
    r_D=0.03,
    r_K=0.03,
    r_K_rent=0.03,
    co_K=0.01,
    co_D=0.004,
    ci_K=10,
    discount=0.05,
    EoS=0.85,
):
    """This function is calculates the NPV as a function of a Demand and Capacity Vector

    Args:
        D           Demand Vector                                       np.array
        K           Estimated Capacity Vector                           np.array
        Fth         Forecast time horizon                               int
        dt          Steptime [Years]                                    int
        th          Throughput Capacity per Unit of Capacity            int
        r_D         Revenues per Unit of Demand per Period              float
        r_K         Revenues per Unit of Capacity per Period            float
        r_K_rent    Rental Revenues per Unit of Capacity per Period     float
        co_K        Operational costs per unit of capacity per period   flaot
        co_D        Operational cost per unit of demand per period      float
        ci_K        Installation cost per unit of capacity              float
        discount    Discount factor                                     float
        EoS         Economy of Scale factor                             float

    Returns:
        NPV for given Inputs                                            np.array

    To call this Function use following syntax:
        Flex_NPV_Calculation(D, K, Fth, dt, r_D, r_K, co_K, co_D, ci_K, discount, EoS)
    """
    Fth += 1
    # Creation of a time vector
    t = np.arange(1, Fth, dt)

    # Creation of a Capacity Change Vector
    deltaK0 = np.diff(K, axis=1)
    # Setting the initial Value of the Change Vector to Zero
    deltaK = np.insert(deltaK0, 0, 0, axis=1)

    # If loop when the Demand Matrix is only a Vector
    if D.ndim == 1:
        D = D.reshape(1, -1)
    else:
        D = D
    # Calculate the Difference Matrix
    diff = K - D
    # Initialize an array to store Revenue for each value
    Revenue = np.zeros(D.shape)
    # Initialize an array to store NPV for each row
    NPV = np.zeros(D.shape[0])
    # For loop to iterate over all Scenarios
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            # If condition to account for over- or undercapacity
            if diff[i, j] > 0:
                Revenue[i, j] = (
                    D[i, j] * r_K_rent + K[i, j] * r_K + th * D[i, j] * r_D
                ) - ((K[i, j] - D[i, j]) * r_K_rent + (K[i, j] - D[i, j]) * th * r_K)
            else:
                Revenue[i, j] = K[i, j] * r_K_rent + K[i, j] * r_K + th * D[i, j] * r_D

        # Operationals Costs as a function of Demand and Capacity
        Cost_Ops = co_K * K + co_D * th * D
        # Investment Costs as a function of Capacity and Economys of Scale Factor
        Cost_Investment = ci_K * (np.power(deltaK, EoS))
        # Total Cost as Sum of Operational and Investment Costs
        Cost = Cost_Ops + Cost_Investment
        # Profit as Difference of Revenue and Total Cost
        Profit = Revenue - Cost
        # Discount rate as a function of the Discount Factor and time t
        Discount = 1 / (1 + discount) ** t
        # Present Value as a function of Discount rate and Profit
        Present_Value = Discount * Profit
        # Net Present Value as sum of all Present Values
        NPV[i] = np.sum(Present_Value[i])
    return NPV
