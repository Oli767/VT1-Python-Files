# Import of Packages
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import sys


def Scenario_creation(mu, sigma, Dt0, dt=1, Fth=20, Forecasts=100):
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
    # Adding one to the time horizon and Forecast as Python starts counting from zero
    Fth += 1
    Forecasts += 1
    # Creation of Number of Forecast Vector
    S = list(range(1, Forecasts))
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


def Scenario_plot(
    Scenarios, Fth=20, NoStep=True, Title="Demand Szenarios", label="Passenger Numbers"
):
    """This function is plotting the Scenarios Created in the Scenario function

    Args:
        Scenarios   Szenario Data                       np.array
        Fth         Forecast time horizon               int
        NoStep      Question if Step Plot or not        bool
        Title       Title for the Plot                  str
        ylabel      Y-Axis Description                  str
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

    indices = np.random.choice(Scenarios.shape[0], size=(40))
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


def NPV_Calculation_Fix(
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
        NPV_Calculation_Fix(D, K, Fth, dt, r_D, r_K, co_K, co_D, ci_K, discount, EoS)
    """

    # If loop when the Demand is only a Vector
    if D.ndim == 1:
        D = D.reshape(1, -1)
    else:
        D = D
    # If loop when the Capacity is only a Vector
    if K.ndim == 1:
        K = np.vstack([K] * D.shape[0])
    else:
        K = K

    Fth += 1
    # Creation of a time vector
    t = np.arange(1, Fth, dt)

    # Creation of a Capacity Change Vector
    deltaK0 = np.diff(K, axis=1)
    # Setting the initial Value of the Change Vector to Zero
    deltaK = np.insert(deltaK0, 0, 0, axis=1)

    # Calculate the Difference Matrix
    diff = K - D
    # Initialize an array to store Revenue for each value
    Revenue = np.zeros(D.shape)
    # Initialize an array to store NPV for each row
    NPV = np.zeros(D.shape[0])

    greater_than_zero = np.greater(diff, 0).astype(int)
    less_than_or_equal_zero = np.less_equal(diff, 0).astype(int)

    Revenue1 = greater_than_zero * (
        (D * r_K_rent + K * r_K + th * D * r_D)
        - ((K - D) * r_K_rent + (K - D) * th * r_K)
    )

    Revenue2 = less_than_or_equal_zero * (K * r_K_rent + K * r_K + th * D * r_D)

    Revenue = Revenue1 + Revenue2
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
    # For loop to iterate over all Scenarios
    for i in range(D.shape[0]):
        # Net Present Value as sum of all Present Values
        NPV[i] = np.sum(Present_Value[i])
    return NPV


def Decision_Rule(D, K0, theta, deltaK):
    """This function is creating new Capacity Vectors while considering a decision rule

    Args:
        D               Demand Vector                       np.array
        K0              Initial Capacity                    integer
        theta           Capacity increase vector            list with 3 integers
        deltaK          Capacity Difference Condition       list with 3 integers

    Returns:
        K_Flex          Capacity vector considering a decision rule    np.array

    To call this Function use following syntax:
        Decision_Rule(D, K0, theta)
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
            if (K_Flex[i, j - 1] - D[i, j]) > 0:
                new_capacity = K_Flex[i, j - 1]
            # elif condition to check the severity of the capacity deficit
            elif (K_Flex[i, j - 1] - D[i, j]) == 0:
                new_capacity = K_Flex[i, j - 1] + theta[0]
            # elif condition to check the severity of the capacity deficit
            elif (K_Flex[i, j - 1] - D[i, j]) < -deltaK[0]:
                new_capacity = K_Flex[i, j - 1] + theta[1]
            # elif condition to check the severity of capacity deficit
            elif (K_Flex[i, j - 1] - D[i, j]) < -deltaK[1]:
                new_capacity = K_Flex[i, j - 1] + theta[2]
            # else condition to check the severity of the Capacity deficit
            else:
                new_capacity = K_Flex[i, j - 1] + theta[3]
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
    # Creation of an array with the same shape as D initialized with K0
    K_Flex = np.full(D.shape, K0, dtype=D.dtype)
    # For loop to iterate over all Scenarios
    for i in range(D.shape[0]):
        # For loop to iterate over all values of a Scenario
        for j in range(D.shape[1]):
            # if condition to check for overcapacity
            if (K_Flex[i, j - 1] - D[i, j]) > 0:
                new_capacity = K_Flex[i, j - 1]
            # else condition to check  for undercapacity
            else:
                new_capacity = K_Flex[i, j - 1] + deltaK_Flex
            # changing the Capacity values for the given overcapacity or deficit
            K_Flex[i, j] = new_capacity
    return K_Flex


def CDF_Plot(Vector1, Vector2):
    """This function is Plotting the Cumulative Density Function of the NPVs
    Args:
        Vector1         Traditional Input Vector 1       np.array
        Vector1         Flexible Input Vector 1          np.array

    Returns:
        Plot of all input Vectors in a CDF Graphic

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
        label="Traditional CDF Curve",
        linewidth=2,
        color="green",
        alpha=0.7,
    )

    ax.plot(
        np.sort(Vector2),
        np.arange(1, len(Vector2) + 1) / float(len(Vector2)),
        linestyle="-",
        label="Flexible CDF Curve",
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
        label="Traditional ENPV",
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
        label="Flexible ENPV",
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
    return [percentile_10a, percentile_90a, percentile_10b, percentile_90b]


def Capacity_Vector_Check(Capacity_Vector, Demand_Array):
    """This function is Cecking if the Capacity Vector and Demand Vector Shape Match

    Args:
        Capacity_Vector         np.array
        Demand_Array            np.array
    Returns:
        Printed Messages or Stops Programm from Running

    To Call the Function use following syntax:
        Capacity_Vector_Check(Capacity_Vector, Demand_Array)
    """
    # If loop when the Demand Matrix is only a Vector
    if Demand_Array.ndim == 1:
        Demand_Array = Demand_Array.reshape(1, -1)
    else:
        Demand_Array = Demand_Array

    # Condition to check the Shape
    some_condition = len(Capacity_Vector) == Demand_Array.shape[1]

    # If loop to Check the condition
    if not some_condition:
        print("Error: Capacity Vector Doesn't Match the Demand Array Shape")
        sys.exit()
    # Continue with the rest of your code if the condition is met
    print("Capacity Vector Matches the Demand Array Shape")
