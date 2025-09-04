# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
from matplotlib import cm

tools.mutGaussian.rng = np.random.RandomState(42)
# Define the problem parameters 
n_months = 12 # total months data
storage_max = 8136  # Maximum storage capacity (MCM)
storage_min = 2318  # Minimum storage capacity (MCM)
S_init = 2318  # Initial storage (assumed)
Pmax = 359.8  # MW Maximum power release capacity 
canal_capacity = 335.53  # Maximum irrigation release capacity MCM

# Efficiency parameter
eta = 0.85  # efficiency of turbines Francis and Kaplan

# Data
random.seed(42)
np.random.seed(42)  # For reproducibility
# Demand for irrigation
D_irr = np.array([200.979, 212.076, 256.464, 242.901, 45.621, 61.65, 177.552, 196.047, 225.639, 257.697, 87.543, 113.436])  # Irrigation demand
# Inflow to hirakud dam
Inf = np.array([188.649, 134.397, 94.941, 48.087, 24.660, 1203.408, 7462.116, 12893.480, 8353.575, 2329.137, 589.374, 244.134])  # Inflows
# Rainfall to dam
Rain = np.array([4.01, 14.86, 36.85, 5.50, 0.00, 138.50, 305.97, 408.80, 157.22, 16.05, 0.00, 5.20])  # rainfall
# total inflow
I = Inf + Rain
# evaporation rate in mm/hr
et = np.array([248.38, 299.61, 439.35, 508.61, 530.81, 414.92, 316.03, 306.02, 292.91, 294.98, 245.44, 238.28])  # evaporation rate in mm
# rate of change of area with storage 
alph = 1/10600  # 1/mm 
# initial area
A0 = 200000000    # area in m2 at MDDL
Fmin = 2318  # Minimum flood storage capacity MCM
min_head = 179.830  # m
# tail elevation to calculate net head
H_tail = 151  # m
# minimum industrial allocation
ind_min = 0   # MCM
# maximum industrial allocation
ind_max = 429.23  # MCM
Dem_ir_max = np.max(D_irr)  # max_irrigation demand
Dem_ir_min=np.min(D_irr)
pow_min=47 #based on research paper
pow_max=650 ##based on research paper
# head calculation function
def calculate_head(S):
    # Assume head varies with storage using power law
    Smin = 1814.976  # MCM
    Smax = 7190.856  # MCM
    Hmin = 179.83    # m
    Hmax = 192.024   # m
    
    # Safely handle the power law to avoid negative values
    if S <= Smin:
        H_res = Hmin
    else:
        # Fit power law: H = Hmin + C*(S-Smin)^n
        H_res = Hmin + 0.0025*((S-Smin)**0.52)
    
    H_net = H_res - H_tail
    return max(H_net, 0)  # Ensure head is non-negative

# max water release
max_water_release = Pmax / (0.00378 * eta * (min_head-H_tail))

# Function to calculate storage trajectory and constraints
def calculate_storage_and_constraints(x_irr, x_pow, x_ind):
    # Calculate total release
    R = x_irr + x_pow + x_ind
    
    # storage trajectory
    S = np.zeros(n_months+1)
    S[0] = S_init
    
    # Track spill for each month
    spill = np.zeros(n_months)
    
    # Track constraint violations
    is_feasible = True
    penalty = 0
    
    # Check constraints
    for t in range(n_months):
        # Update storage
        S_tentative = float((S[t]*(1-0.5*alph*et[t]) + I[t] - (et[t]*A0)/pow(10,9) - R[t])/(1+0.5*alph*et[t]))
        
        # Calculate spill if storage exceeds maximum
        if S_tentative > storage_max:
            spill[t] = S_tentative - storage_max
            S[t+1] = storage_max
            penalty += 1000  # penalty for spill
            
        # Enforce minimum storage constraint 
        elif S_tentative < storage_min:
            # Calculate maximum allowable release to maintain min storage
            max_allowable_release = S[t]*(1-0.5*alph*et[t]) + I[t] - (et[t]*A0)/pow(10,9) - storage_min*(1+0.5*alph*et[t])
            
            # If not enough water, enforce minimum storage anyway
            if max_allowable_release < 0:
                max_allowable_release = 0
            
            # Apply heavy penalty for violating min storage
            penalty += 1000000
            is_feasible = False
            
            # Set storage to minimum
            S[t+1] = storage_min
        else:
            spill[t] = 0
            S[t+1] = S_tentative
        
        # Release non-negativity constraint
        if any(r < 0 for r in [x_irr[t], x_pow[t], x_ind[t]]):
            penalty += 5000
            is_feasible = False
        
        # Power generation capacity constraint
        H_t = calculate_head(S[t])
        if 0.00378 * eta * H_t * x_pow[t] > Pmax:
            penalty += 5000
            is_feasible = False
            
        if t == 6:  # for august month
            if storage_max - S[t+1] < Fmin:
                 penalty += 5000  # Penalty for violating August constraint
                 is_feasible = False
                 
        if t in [0, 1, 2, 3, 4, 5]:    # in non monsoon months when storage goes to low
             if S[t+1] < storage_min + 1000:
                 penalty += 10000
                 is_feasible = False
                 
        # Canal capacity constraint
        if x_irr[t] > canal_capacity:
            penalty += 5000
            is_feasible = False
            
        # Release should not be more than demand     
        if x_irr[t] > Dem_ir_max:
            penalty += 5000
            is_feasible = False
    
    # End storage constraint (13th month storage ≥ initial storage)
    if S[-1] < S_init:
        penalty += 5000
        is_feasible = False
    
    # Calculate hydropower for all months
    hydropower = np.array([0.00378 * eta * calculate_head(S[t]) * x_pow[t] for t in range(n_months)])
    
    return S, spill, hydropower, penalty, is_feasible

# Set up the optimization problem using DEAP
def setup_optimization(w1, w2, w3):
    # Reset creator if already exists
    if 'FitnessMin' in creator.__dict__:
        del creator.FitnessMin
    if 'Individual' in creator.__dict__:
        del creator.Individual
    
    # Create fitness and individual types
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Single objective
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    
    # Each individual consists of 36 values: irrigation, power, industrial releases for 12 months
    def create_individual():
        # Initialize with random values within reasonable bounds
        x_irr = [random.uniform(Dem_ir_min, min(canal_capacity, Dem_ir_max)) for _ in range(n_months)]
        x_pow = [random.uniform(pow_min,pow_max) for _ in range(n_months)]  # power release limit as per data
        x_ind = [random.uniform(ind_min, ind_max) for _ in range(n_months)]  # bounds for industrial release
        return x_irr + x_pow + x_ind
    
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Define the evaluation function
    def evaluate(individual):
        x_irr = np.array(individual[:n_months])
        x_pow = np.array(individual[n_months:2*n_months])
        x_ind = np.array(individual[2*n_months:])
        
        # Calculate storage and check constraints
        S, spill, hydropower, penalty, _ = calculate_storage_and_constraints(x_irr, x_pow, x_ind)
        
        # Objective functions
        f1 = np.sum((D_irr - x_irr) ** 2)  # Minimize irrigation deficit squared
        f2 = np.sum(hydropower)  # Maximize total hydropower
        f3 = np.sum(x_ind)  # Minimize industrial supply
        
        # Weighted objective
        obj_value = w1 * f1 - w2 * f2 + w3 * f3 + penalty
        
        # Store individual objective values for later use
        individual.f1 = f1
        individual.f2 = f2
        individual.f3 = f3
        individual.spill = spill
        individual.storage = S
        individual.hydropower = hydropower
        
        return (obj_value,)  # Return single value in a tuple
    
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    return toolbox

# Function to run optimization with specific weights
def optimize_with_weights(w1, w2, w3, generations=50, population_size=100):
    print(f"Optimizing with weights: w1={w1}, w2={w2}, w3={w3}")
    
    # Setup optimization
    toolbox = setup_optimization(w1, w2, w3)
    
    # Create initial population
    pop = toolbox.population(n=population_size)
    
    # Statistics setup
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    
    # Run the genetic algorithm
    result, log = algorithms.eaSimple(pop, toolbox, 
                                    cxpb=0.7,  # Crossover probability
                                    mutpb=0.2,  # Mutation probability
                                    ngen=generations, 
                                    stats=stats,
                                    verbose=True)
    
    # Get the best individual
    best_ind = tools.selBest(result, k=1)[0]
    
    # Extract the decision variables
    x_irr = np.array(best_ind[:n_months])
    x_pow = np.array(best_ind[n_months:2*n_months])
    x_ind = np.array(best_ind[2*n_months:])
    
    # Calculate storage trajectory and hydropower
    S, spill, hydropower, _, _ = calculate_storage_and_constraints(x_irr, x_pow, x_ind)
    
    # convert them into lists for simplicity
    x_irr = list(x_irr)
    x_pow = list(x_pow)
    x_ind = list(x_ind)
    S = list(S)
    spill = list(spill)
    hydropower = list(hydropower)
    
    return {
        'weights': (w1, w2, w3),
        'best_solution': best_ind,
        'irrigation': x_irr,
        'power': x_pow,
        'industrial': x_ind,
        'storage': S,
        'spill': spill,
        'hydropower': hydropower,
        'total_hydropower': sum(hydropower),
        'objective_values': (best_ind.f1, best_ind.f2, best_ind.f3),
        'fitness': best_ind.fitness.values[0],
        'population': result
    }

def generate_pareto_front(population_size=200, generations=50):
    # Set up multi-objective optimization
    if 'FitnessMulti' in creator.__dict__:
        del creator.FitnessMulti
    if 'Individual' in creator.__dict__:
        del creator.Individual
    
    # Proper weights for minimizing f1, maximizing f2, minimizing f3
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0, -1.0)) 
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    
    toolbox = base.Toolbox()
    
    # Define the structure of an individual
    def create_individual():
        x_irr = [random.uniform(0, min(canal_capacity, Dem_ir_max)) for _ in range(n_months)]
        x_pow = [random.uniform(47, 650) for _ in range(n_months)]
        x_ind = [random.uniform(ind_min, ind_max) for _ in range(n_months)]
        return x_irr + x_pow + x_ind
    
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Define the evaluation function for multi-objective optimization
    def evaluate_multi(individual):
        x_irr = np.array(individual[:n_months])
        x_pow = np.array(individual[n_months:2*n_months])
        x_ind = np.array(individual[2*n_months:])
        
        # Calculate storage and check constraints
        S, spill, hydropower, penalty, is_feasible = calculate_storage_and_constraints(x_irr, x_pow, x_ind)
        
        # Calculate pure objective functions
        f1 = float(np.sum((D_irr - x_irr) ** 2))  # Convert to float
        f2 = float(np.sum(hydropower))  # Convert to float - Total annual hydropower
        f3 = float(np.sum(x_ind))  # Convert to float
        
        # Store the raw objectives and solution details
        individual.raw_objectives = (f1, f2, f3)
        individual.is_feasible = is_feasible
        individual.irrigation = list(x_irr)
        individual.power_release = list(x_pow)
        individual.industrial = list(x_ind)
        individual.storage = list(S)
        individual.spill = list(spill)
        individual.hydropower = list(hydropower)
        individual.total_hydropower = f2  # Store the total hydropower
        
        # For NSGA-II, apply penalties if infeasible, but in a way that preserves Pareto dominance
        if not is_feasible:
            return f1 + 1e6, -1e6, f3 + 1e6  # Apply penalties for constraint violations
        else:
            return f1, f2, f3  # Return raw objectives if feasible
    
    toolbox.register("evaluate", evaluate_multi)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    
    #initial population
    pop = toolbox.population(n=population_size)
    
    # Run NSGA-II algorithm
    algorithms.eaMuPlusLambda(pop, toolbox, 
                             mu=population_size,
                             lambda_=population_size,
                             cxpb=0.7,
                             mutpb=0.2,
                             ngen=generations,
                             verbose=True)
    
    # Extract the Pareto front
    pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    
    # Filter out infeasible solutions if any exist
    feasible_front = [ind for ind in pareto_front if ind.is_feasible]
    
    print(f"Total solutions in Pareto front: {len(pareto_front)}")
    print(f"Feasible solutions in Pareto front: {len(feasible_front)}")
    
    # Use feasible solutions if available, otherwise use all solutions
    final_front = feasible_front if feasible_front else pareto_front
    
    # Extract raw objective values for Pareto front (without penalties)
    pareto_objectives = [ind.raw_objectives for ind in final_front]
    
    # Print range of hydropower values for debugging
    hp_values = [obj[1] for obj in pareto_objectives]
    print(f"Pareto front hydropower range: {min(hp_values):.2f} - {max(hp_values):.2f} MW")
    
    return final_front, pareto_objectives

def plot_pareto_front(pareto_front, pareto_objectives):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract raw objective values
    f1_values = [obj[0] for obj in pareto_objectives]  # Irrigation deficit squared
    f2_values = [obj[1] for obj in pareto_objectives]  # Total annual hydropower (sum of all months)
    f3_values = [obj[2] for obj in pareto_objectives]  # Industrial supply
    
    print(f"Plotting f2 (hydropower) range: {min(f2_values):.2f} - {max(f2_values):.2f}")
    
    # Create scatter plot
    scatter = ax.scatter(f1_values, f2_values, f3_values, c=f2_values, cmap=cm.viridis, 
                         s=50, alpha=0.7)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Total Annual Hydropower (MW)')  # Clarify this is annual total
    
    # Add labels with clearer descriptions
    ax.set_xlabel('Irrigation Deficit Squared - Minimize')
    ax.set_ylabel('Total Annual Hydropower (MW) - Maximize')  # Clarify this is annual total
    ax.set_zlabel('Industrial Supply (MCM) - Minimize')
    ax.set_title('Pareto Front')
    
    # Set axis ranges to match the values in your comparison table
    ax.set_xlim([min(f1_values) * 0.95, max(f1_values) * 1.05])
    ax.set_ylim([min(f2_values) * 0.95, max(f2_values) * 1.05])
    ax.set_zlim([min(f3_values) * 0.95, max(f3_values) * 1.05])
    
    plt.tight_layout()
    plt.show()

# Function to pick representative solutions from the Pareto front
def select_representatives(pareto_front, pareto_objectives, num_representatives=3):
    # Get min and max values for each objective
    f1_min = min(obj[0] for obj in pareto_objectives)
    f1_max = max(obj[0] for obj in pareto_objectives)
    f2_min = min(obj[1] for obj in pareto_objectives)
    f2_max = max(obj[1] for obj in pareto_objectives)
    f3_min = min(obj[2] for obj in pareto_objectives)
    f3_max = max(obj[2] for obj in pareto_objectives)
    
    representatives = []
    
    # Pick solution with best f1 (min irrigation deficit)
    best_f1_idx = np.argmin([obj[0] for obj in pareto_objectives])
    representatives.append(pareto_front[best_f1_idx])
    
    # Pick solution with best f2 (max hydropower)
    best_f2_idx = np.argmax([obj[1] for obj in pareto_objectives])
    representatives.append(pareto_front[best_f2_idx])
    
    # Pick solution with best f3 (min industrial supply)
    best_f3_idx = np.argmin([obj[2] for obj in pareto_objectives])
    representatives.append(pareto_front[best_f3_idx])
    
    # Pick a balanced solution (closest to the normalized center of the Pareto front)
    normalized_objectives = []
    for obj in pareto_objectives:
        norm_f1 = (obj[0] - f1_min) / (f1_max - f1_min) if f1_max > f1_min else 0
        norm_f2 = 1 - (obj[1] - f2_min) / (f2_max - f2_min) if f2_max > f2_min else 0  # Invert since we maximize f2
        norm_f3 = (obj[3] - f3_min) / (f3_max - f3_min) if f3_max > f3_min else 0
        normalized_objectives.append((norm_f1, norm_f2, norm_f3))
    
    #solution closest to the center (0.5, 0.5, 0.5)
    distances = [np.sqrt((obj[0]-0.5)**2 + (obj[1]-0.5)**2 + (obj[2]-0.5)**2) for obj in normalized_objectives]
    balanced_idx = np.argmin(distances)
    representatives.append(pareto_front[balanced_idx])
    
    return representatives[:num_representatives]  # Return requested number of representatives

# Function to plot results for a single solution
def plot_results(result, title_suffix=""):
    fig, axs = plt.subplots(5, 1, figsize=(12, 20))
    
    months = range(1, n_months+1)
    
    # Plot releases
    axs[0].bar(months, result['irrigation'], width=0.25, label='Irrigation')
    axs[0].bar([m + 0.25 for m in months], result['power'], width=0.25, label='Power')
    axs[0].bar([m + 0.5 for m in months], result['industrial'], width=0.25, label='Industrial')
    axs[0].plot(months, D_irr, 'o--', color='red', label='Irrigation Demand')
    axs[0].set_xlabel('Month')
    axs[0].set_ylabel('Release (MCM)')
    
    if 'weights' in result:
        w1, w2, w3 = result['weights']
        axs[0].set_title(f'Optimized Releases (w1={w1:.2f}, w2={w2:.2f}, w3={w3:.2f}) {title_suffix}')
    else:
        axs[0].set_title(f'Optimized Releases {title_suffix}')
    axs[0].legend()
    
    # Plot storage trajectory
    axs[1].plot(range(n_months+1), result['storage'], 'o-', label='Storage')
    axs[1].axhline(y=storage_max, color='r', linestyle='--', label='Max Storage')
    axs[1].axhline(y=storage_min, color='r', linestyle='-.', label='Min Storage')
    axs[1].set_xlabel('Month')
    axs[1].set_ylabel('Storage (MCM)')
    axs[1].set_title('Storage Trajectory')
    axs[1].legend()
    
    # Plot irrigation demand vs. supply
    axs[2].plot(months, D_irr, 'o-', label='Irrigation Demand')
    axs[2].plot(months, result['irrigation'], 's-', label='Irrigation Supply')
    axs[2].set_xlabel('Month')
    axs[2].set_ylabel('Water (MCM)')
    axs[2].set_title('Irrigation Demand vs Supply')
    axs[2].legend()
    
    # Plot hydropower generation
    axs[3].plot(months, result['hydropower'], 'o-', label='Hydropower')
    axs[3].set_xlabel('Month')
    axs[3].set_ylabel('Hydropower Generation (MW)')
    axs[3].set_title('Monthly Hydropower Generation')
    axs[3].legend()
    
    # Plot spill
    axs[4].plot(months, result['spill'], 'o-', label='Spill')
    axs[4].set_xlabel('Month')
    axs[4].set_ylabel('Spill (MCM)')
    axs[4].set_title('Monthly Spill')
    axs[4].legend()
    
    plt.tight_layout()
    plt.show()

#extract solution from Pareto front individual
def extract_pareto_solution(pareto_ind):
    result = {
        'irrigation': pareto_ind.irrigation,
        'power': pareto_ind.power_release,
        'industrial': pareto_ind.industrial,
        'storage': pareto_ind.storage,
        'spill': pareto_ind.spill,
        'hydropower': pareto_ind.hydropower,
        'total_hydropower': pareto_ind.total_hydropower,
        'objective_values': pareto_ind.raw_objectives
    }
    return result

# Function to plot comparison 
def plot_weight_comparison(results_list):
    fig, axs = plt.subplots(4, 1, figsize=(12, 20))
    
    months = range(1, n_months+1)
    
    # Plot for irrigation
    for i, result in enumerate(results_list):
        if 'weights' in result:
            w1, w2, w3 = result['weights']
            label = f'w1={w1:.2f}, w2={w2:.2f}, w3={w3:.2f}'
        else:
            label = f'Pareto Solution {i+1}'
        axs[0].plot(months, result['irrigation'], 'o-', label=label)
    
    axs[0].plot(months, D_irr, 'k--', label='Demand')
    axs[0].set_xlabel('Month')
    axs[0].set_ylabel('Release (MCM)')
    axs[0].set_title('Irrigation Releases Comparison')
    axs[0].legend()
    
    # Plot for power
    for i, result in enumerate(results_list):
        if 'weights' in result:
            w1, w2, w3 = result['weights']
            label = f'w1={w1:.2f}, w2={w2:.2f}, w3={w3:.2f}'
        else:
            label = f'Pareto Solution {i+1}'
        axs[1].plot(months, result['power'], 'o-', label=label)
    
    axs[1].set_xlabel('Month')
    axs[1].set_ylabel('Release (MCM)')
    axs[1].set_title('Power Releases Comparison')
    axs[1].legend()
    
    # Plot for industrial
    for i, result in enumerate(results_list):
        if 'weights' in result:
            w1, w2, w3 = result['weights']
            label = f'w1={w1:.2f}, w2={w2:.2f}, w3={w3:.2f}'
        else:
            label = f'Pareto Solution {i+1}'
        axs[2].plot(months, result['industrial'], 'o-', label=label)
    
    axs[2].set_xlabel('Month')
    axs[2].set_ylabel('Release (MCM)')
    axs[2].set_title('Industrial Releases Comparison')
    axs[2].legend()
    
    # Plot for hydropower
    for i, result in enumerate(results_list):
        if 'weights' in result:
            w1, w2, w3 = result['weights']
            label = f'w1={w1:.2f}, w2={w2:.2f}, w3={w3:.2f}'
        else:
            label = f'Pareto Solution {i+1}'
        axs[3].plot(months, result['hydropower'], 'o-', label=label)
    
    axs[3].set_xlabel('Month')
    axs[3].set_ylabel('Power (MW)')
    axs[3].set_title('Hydropower Generation Comparison')
    axs[3].legend()
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Weight combinations 
    weight_combinations = [
        (0.33, 0.33, 0.33),  # Equal weights
        (0.7, 0.1, 0.2),     # Balanced with irrigation focus
        (0.3, 0.4, 0.3),     # Balanced with hydropower focus
        (0.2, 0.1, 0.7)      # Industrial focus
    ]
    
    # Run optimization for each weight combination
    results = []
    for weights in weight_combinations:
        result = optimize_with_weights(*weights, generations=30, population_size=200)
        results.append(result)
        plot_results(result, f"Weight Set: {weights}")
    
    # Plot comparison of different weight scenarios
    plot_weight_comparison(results)
    
    # comparison table for weighted objectives
    print("\n Weighted Optimization Results:")
    print("-" * 120)
    print(f"{'Weights (w1,w2,w3)':<25} {'Irrigation deficit²':<20} {'Total Hydropower (MW)':<25} {'Industrial supply':<20} {'Total HP (MW)':<15}")
    print("-" * 120)
    
    for result in results:
        weight_str = f"({result['weights'][0]:.2f}, {result['weights'][1]:.2f}, {result['weights'][2]:.2f})"
        print(f"{weight_str:<25} {result['objective_values'][0]:<20.2f} {result['objective_values'][1]:<25.2f} {result['objective_values'][2]:<20.2f} {result['total_hydropower']:<15.2f}")
    
    # Generate Pareto front
    print("\nGenerating Pareto front...")
    pareto_front, pareto_objectives = generate_pareto_front(population_size=150, generations=40)
    
    # Plot Pareto front
    plot_pareto_front(pareto_front, pareto_objectives)
   

