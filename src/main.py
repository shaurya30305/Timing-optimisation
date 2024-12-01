import random
import copy
import math

class defaultdict:
    def __init__(self, default_factory):
        if default_factory not in (list, set):
            raise TypeError("default_factory must be either list or set")
        self.default_factory = default_factory
        self.store = {}

    def __getitem__(self, key):
        if key not in self.store:
            self.store[key] = self.default_factory()  # Create a new list or set
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __contains__(self, key):
        return key in self.store

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def items(self):
        return self.store.items()

    def keys(self):
        return self.store.keys()

    def values(self):
        return self.store.values()

    def clear(self):
        self.store.clear()

    def copy(self):
        new_dict = defaultdict(self.default_factory)
        new_dict.store = self.store.copy()
        return new_dict

    def update(self, other):
        for key, value in other.items():
            self[key] = value

    def __repr__(self):
        return f'defaultdict({self.default_factory.__name__}, {self.store})'
class MyError(Exception):
     def __init__(self, message):
         self.message = message
def generate_initial_stacked_config(gates,att=0):
    max_width = att+max([gate.width for gate in gates])
    current_y = 0
    current_x=0
    c = gates
    maxyinrow = 0
    for gate in c:
        gate.x = current_x
        gate.y = current_y
        maxyinrow = max(maxyinrow, gate.height)
        current_x += gate.width
        if current_x >= max_width:
            current_x = 0
            current_y += maxyinrow
            maxyinrow = 0
    return c

class Gate:
    def __init__(self, id, width, height, delay, x=0, y=0, pins=None):
        self.id = id
        self.width = width
        self.height = height
        self.delay = delay
        self.x = x
        self.y = y
        self.pins = pins if pins is not None else []

    def __repr__(self):
        return f"Gate(id={self.id}, width={self.width}, height={self.height}, delay={self.delay}, x={self.x}, y={self.y}, pins={self.pins})"

def construct_graph(gates, connections, wire_delay_factor):
    graph = defaultdict(list)
    gate_dict = {gate.id: gate for gate in gates}

    connection_groups = defaultdict(set)
    for g1, p1, g2, p2 in connections:
        
        connection_groups[g1, p1].add((g2, p2))

    
    processed_pins = set()

    def find_connected_pins(start_pin, group):
        if start_pin in processed_pins:
            return
        group.append(start_pin)
        processed_pins.add(start_pin)
        for connected_pin in connection_groups[start_pin]:
            find_connected_pins(connected_pin, group)
    for key in list(connection_groups.keys()):
        g1 =[]
        find_connected_pins(key,g1)
        if key[0] in [o[0] for o in g1][1:]:
            
            raise MyError('Cycle found!')
    for key in list(connection_groups.keys()):
        
        group=connection_groups[key]
        xcoords = []
        ycoords=[]
        g1,p1=key[0],key[1]
        gate1 = gate_dict[g1]
        x1 = gate1.x + gate1.pins[p1][0]
        y1 = gate1.y + gate1.pins[p1][1]
        xcoords.append(x1)
        ycoords.append(y1)
        for k in group:
            
            g2,p2 = k[0],k[1]
            
            gate2 = gate_dict[g2]
            xcoords.append(gate2.x + gate2.pins[p2][0])
            ycoords.append(gate2.y+ gate2.pins[p2][1])
        delay = (max(xcoords)-min(xcoords))+(max(ycoords)-min(ycoords))
        for k in group:
            g2,p2 = k[0],k[1]    
            source = f'g{g1}.p{p1}'
            dest = f'g{g2}.p{p2}'
            graph[source].append((dest,delay*wire_delay_factor))
    
        


    for gate in gates:
        for i in range(len(gate.pins)):
            for j in range(i + 1, len(gate.pins)):
                source = f'g{gate.id}.p{i}'
                dest = f'g{gate.id}.p{j}'
                graph[source].append((dest, gate.delay))
                graph[dest].append((source, gate.delay))

    return graph

def find_min_delay_paths(primary_inputs, primary_outputs, gates, connections, graph, wire_delay_factor):
    
    def dfs(graph, start, end):
        stack = [(start, [start], 0)]  # (current_node, path, total_delay)
        all_paths = []
        max_delays = {}  # Keep track of the maximum delay for each node

        while stack:
            current_node, path, total_delay = stack.pop()

            if current_node == end:
                all_paths.append((path, total_delay))
                continue

            if current_node not in graph:
                continue

            for neighbor, segment_delay in graph[current_node]:
                if neighbor not in path:  # Avoid cycles
                    new_delay = total_delay + segment_delay
                    
                    # Update max_delays to avoid unnecessary revisits
                    if neighbor not in max_delays or new_delay > max_delays[neighbor]:
                        max_delays[neighbor] = new_delay
                        stack.append((neighbor, path + [neighbor], new_delay))

        return all_paths

    all_path_delays = []
    for pi in primary_inputs:
        for po in primary_outputs:
            paths = dfs(graph, pi,po)
           
            for j in range(len(paths)):
                i=0
                path,delay = paths[j][0], paths[j][1]
                s=[]
                while i < len(path):
                    s.append(path[i])
                    if i < len(path) - 1 and path[i][1] == path[i+1][1]:
                        k = i + 1
                        while k < len(path) - 1 and path[i][1] == path[k+1][1]:
                            k += 1
                            id1 = int(path[i][1])
                            gt = [h for h in gates if h.id == id1]
                            gtt = gt[0]
                            delay -= gtt.delay
                        if k > i + 1:
                            s.append(path[k])
                            i = k+ 1
                        else:
                            i += 1
                    else:
                        i += 1
                paths[j]=(s,delay)
            
            if paths:
                max_path, max_delay = max(paths, key=lambda x: x[1])
                all_path_delays.append((f"{pi} -> {po}", max_path, max_delay))

    all_path_delays.sort(key=lambda x: x[2], reverse=True)
  
    return all_path_delays
    

def find_critical_path_delays(gates, connections, wire_delay_factor):
    graph = construct_graph(gates, connections, wire_delay_factor)
    connection_inputs = set()
    connection_outputs = set()

    for g1, p1, g2, p2 in connections:
        
        connection_inputs.add((g1,p1))
        connection_outputs.add((g2,p2))

    primary_inputs = []
    primary_outputs = []

    for gate in gates:
        for i in range(len(gate.pins)):
            if (gate.id, i) not in connection_inputs and (gate.id,i) not in connection_outputs:
                if gate.pins[i][0] == 0:
                    primary_inputs.append(f'g{gate.id}.p{i}')
                else:
                    primary_outputs.append(f'g{gate.id}.p{i}')
    
    critical_paths = find_min_delay_paths(primary_inputs, primary_outputs, gates, connections, graph, wire_delay_factor)
    
    critical_path_delay = critical_paths[0][2] if critical_paths else 0

    return critical_path_delay, critical_paths[0]



def parse_input(file_path):
    gates = []
    connections = []
    wire_delay_factor = None
    
    def parse_gate_pin(gate_pin_str):
        gate_id = int(gate_pin_str.split('.')[0][1:])
        pin_id = int(gate_pin_str.split('.')[1][1:])
        return gate_id, pin_id - 1

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0].startswith('g'):
                gate_id = int(parts[0][1:])
                width, height, delay = map(int, parts[1:4])
                gates.append(Gate(gate_id, width, height, delay))
            elif parts[0] == 'pins':
                gate_id = int(parts[1][1:])
                pins = [(int(parts[i]), int(parts[i+1])) for i in range(2, len(parts), 2)]
                gates[gate_id-1].pins = pins
            elif parts[0] == 'wire':
                g1, p1 = parse_gate_pin(parts[1])
                g2, p2 = parse_gate_pin(parts[2])
                connections.append((g1, p1, g2, p2))
            elif parts[0] == 'wire_delay':
                wire_delay_factor = float(parts[1])

    if wire_delay_factor is None:
        raise ValueError("Wire delay factor not found in input file")

    return gates, connections, wire_delay_factor

def next_iteration(gates, swap_prob=0.5,prob=0.3):
    max_attempts = 1000
    
    for _ in range(max_attempts):
        new_gates = copy.deepcopy(gates)
        i = random.randrange(len(new_gates))
        gate_to_move = new_gates[i]
        moves = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        
        for _ in range(len(moves)):
            dx, dy = random.choice(moves)
            moves.remove((dx, dy))
            
            new_x = gate_to_move.x + dx
            new_y = gate_to_move.y + dy
            
            if new_x >= 0 and new_y >= 0:
                gate_to_move.x = new_x
                gate_to_move.y = new_y
                break
        
        if random.random() <= swap_prob:
            i, j = random.sample(range(len(new_gates)), 2)
            new_gates[i], new_gates[j] = new_gates[j], new_gates[i]
        if random.random()<=prob:
            w1 = max([gate.width for gate in gates])
            w2 = sum([gate.width for gate in gates])
            att1 = random.randint(1,w2-w1)
            new_gates = generate_initial_stacked_config(new_gates,att1)
        
        if not check_overlap(new_gates):
            return new_gates
    
    return copy.deepcopy(gates)

g,c,w = parse_input('input.txt')

def check_overlap(gates):
    def is_overlapping(gate1, gate2):
        if (gate1.x + gate1.width <= gate2.x) or (gate2.x + gate2.width <= gate1.x):
            return False
        
        if (gate1.y + gate1.height <= gate2.y) or (gate2.y + gate2.height <= gate1.y):
            return False
        
        return True

    for i in range(len(gates)):
        for j in range(i + 1, len(gates)):
            if is_overlapping(gates[i], gates[j]):
                return True
    
    return False

def simulated_annealing(gates, connections, wire_delay_factor, initial_temp=1000, cooling_rate=0.99, min_temp=1, iterations=5):
    def evaluate_config(config):
        path_delays = find_critical_path_delays(config, connections, wire_delay_factor)
        return path_delays[0]

    current_config = gates
    current_cost = evaluate_config(current_config)
    best_config = copy.deepcopy(current_config)
    best_cost = current_cost
    temperature = initial_temp

    while temperature > min_temp:
        for i in range(iterations):
            new_config = next_iteration(current_config)
            new_cost = evaluate_config(new_config)

            if new_cost < current_cost or random.random() < math.exp((current_cost - new_cost) / temperature):
                current_config = new_config
                current_cost = new_cost

                if current_cost < best_cost:
                    best_config = copy.deepcopy(new_config)
                    best_cost = current_cost

        temperature *= cooling_rate

    best_path_delays,best_path = find_critical_path_delays(best_config, connections, wire_delay_factor)

    return best_config, best_path_delays, best_path
g,c,w=parse_input('input.txt')
def main(f):
    random.seed(2823498324)
    g,c,w=parse_input(f)
    g=generate_initial_stacked_config(g)
    bc,d1, d = simulated_annealing(g,c,w)
    
    x = max([gate.x+gate.width for gate in bc]) - min([gate.x for gate in bc])
    y = max([gate.y+gate.height for gate in bc]) - min([gate.y for gate in bc])
    f = open('output.txt','w')
    f.write('bounding_box {} {}\n'.format(x,y))
    f.write('critical_path ')
    
    for i in d[1]:
        gate_id, pin_id = i.split('.p')
        pin_id = int(pin_id) + 1  
        f.write('g{}.p{} '.format(gate_id[1:], pin_id))
    f.write(f'\ncritical_path_delay {int(d1)}\n')
    for gate in bc:
        f.write('g{} {} {}\n'.format(gate.id,gate.x,gate.y))
    f.close()
main()