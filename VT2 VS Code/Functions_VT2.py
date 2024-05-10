# Import of Packages
import math
import numpy as np
import matplotlib.pyplot as plt
import sys


# def Scenario_creation(mu, sigma, Dt0, dt=1, Fth=20, Forecasts=100):
#     """This function is calculating a denfined numer of Scenarios

#     Args:
#         mu          Mean Percentage Growth mu                       float
#         sigma       Standart Deviation of Percentage Growth         float
#         Dt0         Initial Demand t0                               int
#         dt          Duration of Delta t in Years                    int
#         Fth         Forecast time horizon                           int
#         Forecasts   # number of Forecasts (+1 case we start at 0)   int
#     Returns:
#         Szenarios                                                   np.array

#     To Call the Function use following syntax:
#         Scenario_creation(mu, sigma, Dt0, dt, Fth, Forecasts)
#     """
#     # Adding one to the time horizon and Forecast as Python starts counting from zero
#     Fth += 1
#     Forecasts += 1
#     # Creation of Number of Forecast Vector
#     S = list(range(1, Forecasts))
#     # Creation of a time length of the Scenario Vectors
#     S2 = list(range(1, Fth))


#     # Initialise a Scenarios Vector
#     Szenarios = []
#     # For loop to iterate over all Forecasts
#     for i in S:
#         # Add Demand at t0 to the Vector
#         D = [Dt0]
#         # Second for loop to iterate over the time length of the Scenarios
#         for j in S2:
#             # Random number for Spread of the Scenario
#             randomrange = np.random.normal(0, 1)
#             Szenario = D[j - 1] + (
#                 D[j - 1] * mu * dt + D[j - 1] * sigma * randomrange * math.sqrt(dt)
#             )
#             # Append all individual Demand Curves to the Scenarios
#             D.append(Szenario)
#         # Append all Demand Vectors to the Scenarios Maxtrix
#         Szenarios.append(D)
#     # Change Shape to an Numpy Array
#     Szenarios = np.array(Szenarios)
#     # Get rid of the Initial Demand Value
#     Szenarios = Szenarios[:, 1:]
#     return Szenarios


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

    # Adding one to the time horizon as Python starts counting from zero
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

    # Create an Index Matrix with the Condition for Overcapacity
    greater_than_zero = np.greater(diff, 0).astype(int)
    # Create an Index Matrix with the Condition for Undercapacity
    less_than_or_equal_zero = np.less_equal(diff, 0).astype(int)

    # Calculation of the Revenue with in the Overcapacity Condition
    Revenue1 = greater_than_zero * (
        (D * r_K_rent + K * r_K + th * D * r_D)
        - ((K - D) * r_K_rent + (K - D) * th * r_K)
    )
    # Calculation of the Revenue with in the Undercapacity Condition
    Revenue2 = less_than_or_equal_zero * (K * r_K_rent + K * r_K + th * D * r_D)
    # Combine the two Revenue Matrices
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
        theta           Capacity increase vector            list with 4 integers
        deltaK          Capacity Difference Condition       list with 3 integers

    Returns:
        K_Flex          Capacity vector considering a decision rule    np.array

    To call this Function use following syntax:
        Decision_Rule(D, K0, theta)
    """
    # Check if theta has a length of four values and if all values are of int type
    if len(theta) != 4 or not all(isinstance(value, int) for value in theta):
        raise ValueError(
            "Theta is either not of length 4 or includes values other than integers only"
        )
    # Check if deltaK has a length of three values and if all values are of int type
    if len(deltaK) != 3 or not all(isinstance(value, int) for value in theta):
        raise ValueError(
            "deltaK is either not of length 3 or includes values other than integers only"
        )

    print("Theta and deltaK match the requirements")

    # If loop when the Demand Matrix is only a Vector
    if D.ndim == 1:
        D = D.reshape(1, -1)
    else:
        D = D
    # Create an array of the same shape as D initialized with K0
    K_Flex = np.full(D.shape, K0, dtype=D.dtype)
    # For loop to iterate over all values of a Scenario
    for t in range(D.shape[1]):
        # Calculate the Difference Matrix
        diff = K_Flex[:, t - 1] - D[:, t]
        # Create an Index Matrix with the Condition for Overcapacity
        over_capacity = np.greater(diff, 0).astype(int)
        # Create an Index Matrix with the Condition between 0 and -2 Undercapacity
        under_capacity1 = np.logical_and(
            np.less_equal(diff, 0), np.greater(diff, -deltaK[0])
        ).astype(int)
        # Create an Index Matrix with the Condition between -2 and -5 Undercapacity
        under_capacity2 = np.logical_and(
            np.less_equal(diff, -deltaK[0]), np.greater(diff, -deltaK[1])
        ).astype(int)
        # Create an Index Matrix with the Condition between -2 and -5 Undercapacity
        under_capacity3 = np.logical_and(
            np.less_equal(diff, -deltaK[1]), np.greater(diff, -deltaK[2])
        ).astype(int)
        # Create an Index Matrix with the Condition less or equal than -10 Undercapacity
        under_capacity4 = np.less_equal(diff, -deltaK[2]).astype(int)

        # Update K_Flex for the next iteration
        K_Flex[:, t] = (
            over_capacity * (K_Flex[:, t - 1])
            + under_capacity1 * (K_Flex[:, t - 1] + theta[0])
            + under_capacity2 * ((K_Flex[:, t - 1]) + theta[1])
            + under_capacity3 * ((K_Flex[:, t - 1]) + theta[2])
            + under_capacity4 * ((K_Flex[:, t - 1]) + theta[3])
        )
    return K_Flex


def Decision_Rule_Excel(D, K0=25, deltaK_Flex=5):
    """This function creates new Capacity Vectors while considering a decision rule.

    Args:
        D               Demand Vector               np.array
        K0              Initial Capacity            integer
        deltaK_Flex     Capacity increase vector    int

    Returns:
        K_Flex          Capacity vector considering a decision rule    np.array

    To call this Function use the following syntax:
        Decision_Rule_Excel(D, K0, deltaK_Flex)
    """
    # If loop when the Demand Matrix is only a Vector
    if D.ndim == 1:
        D = D.reshape(1, -1)
    else:
        D = D
    # Creation of an array with the same shape as D initialized with K0
    K_Flex = np.full(D.shape, K0, dtype=D.dtype)

    # For loop to iterate over all values of a Scenario
    for t in range(D.shape[1]):
        # Calculate the Difference Matrix
        diff = K_Flex[:, t - 1] - D[:, t]
        # Create an Index Matrix with the Condition for Overcapacity
        over_capacity = np.greater_equal(diff, 0).astype(int)
        # Create an Index Matrix with the Condition for Undercapacity
        under_capacity = np.less(diff, 0).astype(int)
        # Update K_Flex for the next iteration
        K_Flex[:, t] = over_capacity * (K_Flex[:, t - 1]) + under_capacity * (
            K_Flex[:, t - 1] + deltaK_Flex
        )
    return K_Flex


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
    condition = len(Capacity_Vector) == Demand_Array.shape[1]

    # If loop to Check the condition
    if not condition:
        print("Error: Capacity Vector Doesn't Match the Demand Array Shape")
        print(" Length Capacity Vector = ", len(Capacity_Vector))
        print(" Expected Length (Demand Array Shape) = ", Demand_Array.shape[1])
        sys.exit()
    # Continue with the rest of your code if the condition is met
    print("Capacity Vector Matches the Demand Array Shape")
