
def parse_input(input_str):
    lines = input_str.split('\n')
    weights = []
    clauses = []
    for line in lines:
        if line.startswith('w'):
            weights = [int(x) for x in line.split()[1:-1]]
        elif not line.startswith('c') and not line.startswith('p') and line.strip():
            clauses.append([int(x) for x in line.split()[:-1]])
    return clauses, weights