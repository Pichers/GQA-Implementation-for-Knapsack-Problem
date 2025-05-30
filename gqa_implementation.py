from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit
import numpy as np
import random

# Problem w/ 8 items
weights = [3, 4, 2, 5, 8, 3, 4, 2]
profits = [5, 6, 3, 8, 9, 5, 6, 3]
capacity = 15
num_items = 8

# Initialize qubit states (equal superposition)
qubit_states = [[1/np.sqrt(2), 1/np.sqrt(2)] for _ in range(num_items)]
best_solution = [0] * num_items
best_profit = 0

# Rotation gate
def rotation_gate(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], 
                    [np.sin(theta), np.cos(theta)]])

# Calculate total weight of a solution
def calculate_weight(solution):
    return sum(weights[i] for i in range(len(solution)) if solution[i])

# GQA iterations
for generation in range(10): 
    print(f"\nGeneration {generation}:")
    candidate_solution = []
    
    # Measure qubits to get candidate solution
    for i in range(num_items):
        prob_0 = abs(qubit_states[i][0])**2
        if random.random() < prob_0:
            candidate_solution.append(0)
        else:
            candidate_solution.append(1)
    
    # Repair infeasible solutions
    total_weight = calculate_weight(candidate_solution)
    if total_weight > capacity:
        # Remove items with smallest profit/weight until within capacity
        while total_weight > capacity:
            # Find selected items with profit/weight ratio
            selected_items = [(i, profits[i]/weights[i]) 
                             for i in range(num_items) 
                             if candidate_solution[i]]
            
            if not selected_items:
                break
                
            # Find item with smallest ratio
            min_ratio_index = min(selected_items, key=lambda x: x[1])[0]
            candidate_solution[min_ratio_index] = 0
            total_weight -= weights[min_ratio_index]
    
    # Try to add items to improve solution
    current_profit = sum(profits[i] for i in range(num_items) if candidate_solution[i])
    remaining_capacity = capacity - total_weight
    
    add_candidates = [(i, profits[i]/weights[i]) 
                     for i in range(num_items) 
                     if not candidate_solution[i] and weights[i] <= remaining_capacity]
    
    # Add best candidates first (greedy approach)
    if add_candidates:
        add_candidates.sort(key=lambda x: x[1], reverse=True)
        
        for item_index, _ in add_candidates:
            if weights[item_index] <= remaining_capacity:
                candidate_solution[item_index] = 1
                remaining_capacity -= weights[item_index]
                current_profit += profits[item_index]
    
    # Evaluate profit
    profit = sum(profits[i] for i in range(num_items) if candidate_solution[i])
    if profit > best_profit:
        best_solution, best_profit = candidate_solution.copy(), profit
    
    # Update qubits using rotation gates
    for i in range(num_items):
        if best_solution[i] == 1 and candidate_solution[i] == 0:
            theta = 0.025 * np.pi  # Rotate toward |1⟩
        elif best_solution[i] == 0 and candidate_solution[i] == 1:
            theta = -0.025 * np.pi  # Rotate toward |0⟩
        else:
            theta = 0
        
        U = rotation_gate(theta)
        new_state = U @ np.array(qubit_states[i])
        qubit_states[i] = [new_state[0], new_state[1]]
    
    print(f"Candidate: {candidate_solution}, Profit: {profit}, Weight: {calculate_weight(candidate_solution)}")
    print(f"Qubit Probabilities: {[np.round(abs(qubit_states[i][1])**2, 3) for i in range(num_items)]}")

print(f"\nFinal Best Solution: {best_solution}")
print(f"Selected items: {[i+1 for i in range(num_items) if best_solution[i]]}")
print(f"Total Profit: {best_profit}, Total Weight: {calculate_weight(best_solution)} (Capacity: {capacity})")