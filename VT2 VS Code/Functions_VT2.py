# Import of Packages for Functions
import numpy as np
import random
import matplotlib.pyplot as plt
from deap import base, creator, tools
import itertools


def Generate_scenarios(Param):
    """
    This function is calculating a denfined number of scenarios (Forecasts)
    with a defined (length) number of years (Fth)

    Args:
        Param (dict): Parameter Dictionary

    Returns:
        Demand (ndarray): Demand Matrix

    To call the function use following syntax:
        Scenario_creation(Param)
    """
    # Parameters
    mu = Param["mu"]
    sigma = Param["sigma"]
    Dt0 = Param["Dt0"]
    dt = Param["dt"]
    Fth = Param["Fth"]
    Forecasts = Param["Forecasts"]

    np.random.seed(1)

    # Create arrays for indices
    scenarios = np.arange(0, Forecasts)
    timeseries = np.arange(0, Fth + 1)

    # Random values for spread of the scenario
    random_values = np.random.normal(0, 1, size=(len(scenarios), len(timeseries) - 1))

    # Calculation of the Demand Matrix
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
    """
    This function is plotting any data vector or matrix against the forecast time
    horizon vector Fth, it allows to shows only a selected number (n) of plots

    Args:
        Scenarios (ndarray): Szenario (Plotting) Data
        Fth (int): Forecast Time Horizon
        NoStep (bool): Question if Step Plot or not
        Title (str): Title for the Plot
        label (str): Y-Axis Description
        n (int): Number of Random Selection

    Returns:
        Plot of n Demand Vectors Against the Forecast Time Horizon

    To call the function use following syntax:
        Scenario_plot(Scenarios, Fth, NoStep, Title, label, n)
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


def Capacity(delta_K, Param):
    """
    This function returns the capacity value in matrix format for a given initial
    capacity (K0) and a delta capacity vector (delta_K), copied according to the number
    of scenarios (Forecasts)

    Args:
        K0 (int): Initial Capacity
        delta_K (ndarray): Delta Capacity Vector
        Param (dict): Parameter Dictionary

    Returns:
        K (ndarray): Capacity Matrix

    To call the function use following syntax:
        Capacity(K0, delta_K, Forecasts)
    """
    # Parameter
    K0 = Param["K0"]  # Initial Capacity
    Forecasts = Param["Forecasts"]  # Number of Forecasts

    # Repeat the delta_K vector 'Forecasts' times
    repeated_delta_K = np.repeat(delta_K[np.newaxis, :], Forecasts, axis=0)

    # Create a cumulative sum array starting from initial capacity K0 for each forecast
    K = K0 + np.cumsum(repeated_delta_K, axis=1)

    return K


def Revenue(K, D, r_K, r_K_rent, r_D, condition):
    """
    This Function calculates the revenue with the given inputs of capacity (K), demand
    (D), and further paramters

    Args:
        K (ndarray): Capacity Vector
        D (ndarray): Demand Matrix
        r_K (float): Revenues per Unit of Capacity per Period
        r_K_rent (float): Rental Revenues per Unit of Capacity per Period
        r_D (float): Revenues per Unit of Demand per Period
        condition (int): Condition for Capacity Increase (difference of K and D)

    Returns:
        Total_Revenue (float): Revenue

    To call the function use following syntax:
        Revenue(K, D, r_K, r_K_rent, r_D, condition)
    """
    # Calculating the difference matrix capacity minus demand
    diff = K - D

    # Creating indent matrices for the given conditions
    greater = np.greater(diff, condition).astype(int)
    less_equal = np.less_equal(diff, condition).astype(int)

    # if Overcapacity only amount of Demand can be sold
    rev_overcapacity = greater * (D * r_K + D * r_K_rent + D * r_D)
    # if Undercapacity only available Capacity can be sold
    rev_undercapacity = less_equal * (K * r_K + K * r_K_rent + K * r_D)

    # Summing up all the revenues
    Total_Revenue = rev_overcapacity + rev_undercapacity

    return Total_Revenue


def Cost(K, D, delta_K, co_K, co_D, ci_K, EoS, h, condition):
    """
    This Function calculates the cost with the given inputs of capacity (K), demand
    (D), delta capacity vector (delta_K) and further Paramters

    Args:
        K (ndarray): Capacity Vector
        D (ndarray): Demand Matrix
        delta_K (ndarray): Delta Capacity Vector
        co_K (float): Operating Cost per Unit of Capacity
        co_D (float): Operating Cost per Unit of Demand
        ci_K (float): Installation Cost per Unit of Capaciy
        EoS (float): Economy of Scale Factor
        h (int): -
        condition (int): Condition for Capacity Increase (difference of K and D)

    Returns:
        Total_Cost (float): Cost

    To call the function use following syntax:
        Cost(K, D, delta_K, co_K, co_D, ci_K, EoS, h, condition)
    """
    # Calculating the difference matrix capacity minus demand
    diff = K - D

    # Penalty Cost Overcapacity
    pc_over = 0
    # Penalty Cost Undercapacity
    pc_under = 0
    # Create an Index Matrix for over-, under-, and equalcapacity condition
    cos_overcapacity = np.greater(diff, condition).astype(int)
    cos_undercapacity = np.less(diff, condition).astype(int)
    cos_equalcapacity = np.equal(diff, condition).astype(int)

    Total_Cost = (
        ((ci_K * (delta_K) ** EoS) / h)
        + cos_undercapacity * (pc_under + (co_D * K + co_K * K))
        + cos_overcapacity * (pc_over + (co_D * D + co_K * D))
        + cos_equalcapacity * (co_D * D + co_K * K)
    )

    return Total_Cost


def NPV_calculation(K, D, delta_K, Param, condition):
    """
    This function calculates the net present value by calling the Revenue and Cost
    functions and multiplying it with the discount rate factor

    Args:
        K (ndarray): Capacity Vector
        D (ndarray): Demand Matrix
        delta_K (ndarray): Delta Capacity Vector
        Forecasts (int): Number of Forecasts
        condition (int): Condition for Capacity Increase (difference of K and D)

    Returns:
        NPV (ndarray): Net Present Value

    To call the function use following syntax:
        NPV_calculation(K, D, delta_K, Param, condition)
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
    h = Param["h"]  # h
    Fth = Param["Fth"]  # Time Horizon of Forecasts in Steptime
    dt = Param["dt"]  # Steptime in Years

    # Calling the Revenue function
    Rev = Revenue(K, D, r_K, r_K_rent, r_D, condition)

    # Calling the Cost function
    Cos = Cost(K, D, delta_K, co_K, co_D, ci_K, EoS, h, condition)
    
    # Calculation of the profit
    Profit = Rev - Cos
    
    # Plus one because Python starts at Zero
    t = 1 + np.arange(0, Fth, dt)

    

    # Calulation of the present value with the discount rate factor
    Discount = 1 / (1 + discount) ** t
    Present_value = Profit * Discount

    # Sum up all present values for the net present value
    NPV = np.sum(Present_value, axis=1)

    return NPV


def ENPV_calculation(delta_K, Param, D, condition=0):
    """
    This function calculates the expected net present value by calling the NPV
    Calculation function

    Args:
        delta_K (ndarray): Capacity Increase Vector
        Param (dict): Parameter Dictionary
        D (ndarray): Demand Matrix
        condition (int): Condition for Capacity Increase (difference of K and D)

    Returns:
        ENPV (ndarray): Expected Net Present Value

    To call the function use following syntax:
        ENPV_calculation(delta_K, Param, D, condition)
    """
    # Calling the Capacity function to generate a Capacity Matrix made of delta_K vector
    K = Capacity(delta_K, Param)

    # Calculating the mean of all NPVs to get the ENPV
    ENPV = np.mean(NPV_calculation(K, D, delta_K, Param, condition))

    return ENPV


def GA(Param, D, condition=0):
    """
    This is a Genetic Algorithm seeking to find an optimal delta capacity vector
    (delta_K) to maximise the NPV

    Args:
        Param (dict): Parameter Dictionary
        D (ndarray): Demand Matrix
        condition (int): Condition for Capacity Increase (difference of K and D)

    Returns:
        delta_K (ndarray): Capacity Increase Vector

    To call the function use following syntax:
        GA(Param, D, condition)
    """

    # Define the vector of capacity values allowed to use
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

    # Define an evaluation function
    def evaluate(individual, Param=Param):
        return (ENPV_calculation(individual, Param=Param, D=D, condition=condition),)

    toolbox.register("evaluate", evaluate)

    # Define genetic operators
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Set up the genetic algorithm
    population = toolbox.population(n=Param["population"])
    cxpb, mutpb, ngen = 0.5, 0.2, 10

    # Performance of the evolution
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


def Decision_Rule(K0, D, theta, condition):
    """
    This function creates new delta capacity vector while considering a decision rule

    Args:
        K0 (int): Initial Capacity
        D (ndarray): Demand Matrix
        theta (ndarray): Capacity Change Vector
        condition (int): Condition for Capacity Increase (difference of K and D)

    Returns:
        delta_K_Flex (ndarray): delta capacity vector considering a decision rule

    To call this function use the following syntax:
        Decision_Rule(K0, D, theta, condition)
    """
    # Creation of an array with the same shape as D initialized with initial capacity K0
    K_Flex = np.full(D.shape, K0, dtype=D.dtype)

    # For loop to iterate over all values of a Scenario
    for t in range(1, D.shape[1]):  # Start from t=1
        # Calculate the Difference Matrix
        diff = K_Flex[:, t - 1] - D[:, t]

        # Create an Index Matrix for the condition of over- and undercapacity
        over_capacity = np.greater_equal(diff, condition).astype(int)
        under_capacity = np.less(diff, condition).astype(int)

        # Update K_Flex for the next iteration
        K_Flex[:, t] = over_capacity * K_Flex[:, t - 1] + under_capacity * (
            K_Flex[:, t - 1] + theta
        )

        # Calculation of the delta capacity vector delta_K
        delta_K = np.diff((K_Flex) - K0)
        delta_K_Flex = np.insert(delta_K, 0, 0, axis=1)

    return delta_K_Flex


def Capacity2(K0, delta_K):
    """
    This function returns the Capacity in Matrix format for a given initial
    capacity (K0) and the delta capacity vector (delta_K)

    Args:
        K0 (int): Initial Capacity
        delta_K (ndarray): Delta Capacity Vector

    Returns:
        K (ndarray): Capacity Matrix

    To call this function use the following syntax:
        Capacity2(K0, delta_K)
    """
    # Create a cumulative sum array starting from K0 for each forecast
    K = K0 + np.cumsum(delta_K, axis=1)

    return K


def NPV_Flexible(delta_K, Param, D, condition):
    """
    This function calculates the Net Present Value for the flexible case by calling the
    Capacity and NPV calculation functions

    Args:
        delta_K (ndarray): Delta Capacity Vector
        Param (dict): Parameter Dictionary
        condition (int): Condition for Capacity Increase (difference of K and D)

    Returns:
        NPV (ndarray): Net Present Value for the Flexible Case

    To call this function use the following syntax:
        NPV_Flexible(delta_K, Param)
    """
    # Parameter
    K0 = Param["K0"]  # Initial Capacity

    # Calling the Capacity2 function for the Capacity matrix
    K_Flex = Capacity2(K0, delta_K)

    # Calling the NPV calculation function for the NPV vector
    NPV = NPV_calculation(K_Flex, D, delta_K, Param, condition)

    return NPV


def ENPV_Flexible(theta, condition, Param, D):
    """
    This function calculates the Expected Net Present Value by calling the Decision Rule
    and NPV Flexible functions and using the theta and condition values

    Args:
        theta (ndarray): Capacity increase value
        condition (int): Condition for Capacity increase (difference of K and D)
        Param (dict): Parameter Dictionary
        D (ndarray): Demand Matrix

    Returns:
        ENPV (ndarray): Expected Net Present Value in the Flexible case

    To call this function use the following syntax:
        ENPV_Flexible(theta, condition, Param, D)
    """
    # Parameter
    K0 = Param["K0"]  # Initial Capacity

    # Calling the Decision Rule function for the delta capacity matrix
    delta_K = Decision_Rule(K0, D, theta, condition)

    # Calling the NPV Flexible function for the NPV vector
    NPV = NPV_Flexible(delta_K, Param, D, condition)

    # Calculating the mean of all NPVs to get the ENPV
    ENPV = np.mean(NPV)

    return ENPV


def Optimization(Param, n):
    """
    This function creates a list of tuples consisiting of each pair of theta and
    condition, it reduces the list to a random sample of size n

    Args:
        Param (dict): Parameter Dictionary
        n (int): Sample Size

    Returns:
        optimization_params (list of tuples): List of Theta and Condition Tuple Pairs

    To call this function use the following syntax:
        Optimization(Param, n)
    """
    # Theta
    lower_theta = Param["lower_theta"]
    upper_theta = Param["upper_theta"]
    stepsize_theta = Param["stepsize_theta"]

    # Condition
    lower_cond = Param["lower_cond"]
    upper_cond = Param["upper_cond"]
    stepsize_cond = Param["stepsize_cond"]

    # Creation of a List of Tuples
    condition = np.arange(lower_cond, upper_cond, stepsize_cond)
    theta = np.arange(lower_theta, upper_theta, stepsize_theta)
    optimization_params = list(itertools.product(theta, condition))

    indices = np.random.choice(len(optimization_params), size=n, replace=False)
    optimization_params_sample = [optimization_params[i] for i in indices]

    return optimization_params_sample


def Evaluation(Param, D, n=1000):
    """
    This function first calls the Optimization function to generate a list of tuples
    consisiting of each pair (defined sample size n) of theta and conditon, it then
    continues to evaluates all the tuples by iterating over each pair to find the
    maximum ENPV value

    Args:
        Param (dict): Parameter Dictionary
        D (ndarray): Demand Matrix
        n (int): Sample Size

    Returns:
        max_enpv (int): Maximum value of the ENPV,
        max_theta (int): optimal value of theta,
        max_cond (int): optimal value of the condition

    To call this function use the following syntax:
        Evaluation(Param, D, n)
    """
    # Calling the Optimization function to get the list of tuples
    optimization_params = Optimization(Param, n)

    # Initialize the maximum values
    max_enpv = float("-inf")
    max_theta = None
    max_cond = None

    for theta, condition in optimization_params:
        ENPV = ENPV_Flexible(theta, condition, Param, D)
        if ENPV > max_enpv:
            max_enpv = ENPV
            max_theta = theta
            max_cond = condition

    return max_enpv, max_theta, max_cond


def CDF_Plot(Vector1, Vector2, label1="Vector1", label2="Vector2"):
    """
    This function is Plotting the Cumulative Density Function of the NPVs
    Args:
        Vector1 (ndarray): Input Vector 1
        Vector2 (ndarray): Input Vector 2
        label1 (str): First CDF Curve
        label2 (str): Second CDF Curve
        label3 (str): First ENPV Value
        label4 (str): Second ENPV Value

    Returns:
        Plot of all input Vectors in a CDF Graphic
        + Visualisation of the 10th, 90th Percentile of the Input Vectors

    To call this Function use following syntax:
        CDF_Plot(Vector1, Vector2, label1, label2, label3, label4)
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
        label=label1 + " CDF Curve",
        linewidth=2,
        color="green",
        alpha=0.7,
    )

    ax.plot(
        np.sort(Vector2),
        np.arange(1, len(Vector2) + 1) / float(len(Vector2)),
        linestyle="-",
        label=label2 + " CDF Curve",
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
        label=label1 + " ENPV",
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
        label=label2 + " ENPV",
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


def Dockstands(K, Param):
    """
    This function calculates the Demand for Dockstands given a capacty (K) and
    parameters (Param)

    Args:
        K (ndarray): Capacity Matrix
        Param (dict): Parameter Dictionary

    Returns:
        dockstands (ndarray): Demand for Dockstands

    To call this Function use following syntax:
        Dockstands(K, Param)
    """
    # Parameters
    DHL_factor_20 = Param["DHL_factor_20"]  # Factor to calculate the Demand Hour
    p_Dock = Param["p_dock"]  # Percentage of Pax using Dock Stands
    p_schengen = Param["p_schengen"]  # Percentage of Pax travelling within Schengen
    p_Dok_A_B = Param["p_Dok_A_B"]  # Percentage of Pax travelling from Dock A
    PAXATM = Param["PAXATM"]  # Average passenger no carried per air traffic movement

    DHL = K * DHL_factor_20
    dockstands = np.ceil((DHL * p_Dock * p_schengen * p_Dok_A_B) / PAXATM)

    return dockstands
