# Timing Optimization in Gate Positioning

## Overview
This project implements a circuit layout optimization system focused on minimizing total circuit delay. Given a set of gates with specified dimensions, pin locations, connections, and individual gate delays, the algorithm generates an optimal layout configuration that minimizes the overall timing delay across the entire circuit.

## Problem Description
The algorithm solves the following timing optimization problem:

### Input
- A set of rectangular logic gates (g₁, g₂, ..., gₙ)
- Dimensions (width and height) for each gate
- Pin locations (x, y coordinates) on the boundary of each gate
- Pin-to-pin connections between gates
- Gate delays for each gate in the circuit
- a constant wire delay per unit length
### Constraints
- No overlapping gates allowed
- All connections must be maintained
- Gates must be placed within the specified grid

### Objective
Minimize the total delay path through the circuit, considering:
- Individual gate delays
- Wire delays based on connection lengths
- Critical path optimization

## Algorithm
The project implements simulated annealing. The layout minimises the maximal delay between any primary input and primary output, by convention it is assumed primary
inputs are pins on the left wall of a gate and primary outputs are pins on the right wall. 

Distances are calculated using bounding box method, where from a pin we draw a bounding box to all other pins connected to it, and the wire length to all of them is the 
semiperimeter of this box. A dfs with memoization is used to find the delay of any particular path, which is used by the simulated annealing function to check a particular
configuration


## Installation

### Prerequisites
- Python 3.x

### Setup
1. Clone the repository or download the source files
2. Ensure you have both core files:
   - `src/main.py`: source code
   - `visualiser/visualisation.py`: Layout visualization tool

## Usage

### Running the Program
```bash
# Step 1: Run the main optimization algorithm, change the code so that the main function takes the input file as its input
python main.py  

# Step 2: Visualize the results, pass the input and output files from before as input for this
python visualisation.py  
```

### Input File Format
```ini
[gates]
# gate_name width height delay
g1 4 3 2
pins g1 0 2 4 1
.
.
.
wire_delay 2
wire g1.p2 g2.p1
.
.
.

```

## TESTING AND VALIDATION



1.) Singly connected circuit(checking if the algorithm understands to just cause the pins to overlap)
![image](https://github.com/user-attachments/assets/c27e2ddc-ce4a-460f-9abb-4e2acf2d2eb2)
2.) Multiple connections from a single pin(checking the bounding box method for distance calculation)
![image](https://github.com/user-attachments/assets/020e32cb-2aa8-4451-90b4-dd46a600b9dc)
3.) Circular connection(checking for negative edge weight loops in the dfs algorithm, it just ignores the loop)
![image](https://github.com/user-attachments/assets/7842cd25-c854-48dd-8c89-572ce45faf19)
4.) Correctness check(in this configuration it is possible to cause all connected pins to overlap, checking if the algo finds this layout, I havent run this through the visualiser but we can see that the output delay is just the sum of gate delays)
![image](https://github.com/user-attachments/assets/b9834c4b-6a48-4852-9163-6a2ee9eb7da5)
5.) Densely connected circuit with 25 pins
![image](https://github.com/user-attachments/assets/4ff3cc80-a62c-41f6-8452-2f841341a6a8)


