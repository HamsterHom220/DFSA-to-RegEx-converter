# Sofia Shulyak

# function and class definitions
#------------------------------------------------------------------------------------------
# function for parsing input lines split by '='
def parse(split_str):
    return split_str[1][1:-2].split(',')


# appropriate chars for state names
chars = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
         'w', 'x', 'y', 'z',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
         'V', 'W', 'X', 'Y', 'Z',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
         }
# checks whether a state or an element of the alpha consists of appropriate chars only
def appropriate_name(str, state):
    if state:
        for c in str:
            if c not in chars:
                return False
    else:
        for c in str:
            if (c not in chars) and (c != '_'):
                return False
    return True


# Transition class objects store data in form of s1>token>s2
class Transition:
    def __init__(self, str):
        self.state1, self.token, self.state2 = str.split('>')
    def __eq__(self,other):
        return self.token==other.token
    def __ne__(self, other):
        return self.token!=other.token
    def __lt__(self, other):
        if self.token==other.token:
            return False
        if other.token=="eps":
            return True
        if self.token=="eps":
            return False
        return self.token<other.token
    def __gt__(self, other):
        if self.token==other.token:
            return False
        if other.token=="eps":
            return False
        if self.token=="eps":
            return True
        return self.token>other.token


# sorting function for nearly sorted arrays (taken from https://www.geeksforgeeks.org/insertion-sort/)
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


# Graph class objects represent a graph as Edges List, Adjacency List, Vertices List, and Vertices set
# Supports only nodes with values: 0,1,2,3,...
class Graph:
    # Non-existent vertices are represented as None
    # Vertices List is sorted
    # Examples for Vertices Set = {0,1,2}
    #   Edges List: [(1,0), (0,2), (2,1), (1,2)]
    #   Adjacency List: [[2],[0,2],[1]]

    def __init__(self):
        self.EdgesList = []
        self.EdgesListUndir = []
        self.AdjList = []
        self.AdjListUndir = []
        self.VerticesList = []
        self.VerticesSet = set()

    # function to add a vertex to an instance of graph
    def add_vertex(self,v):
        self.VerticesSet.add(v)
        self.VerticesList.append(v)
        insertion_sort(self.VerticesList)

        n = len(self.AdjList) - 1 # max ind
        while v > n:
            self.AdjList.append([])
            self.AdjListUndir.append([])
            n += 1
        self.AdjList[-1] = []
        self.AdjListUndir[-1] = []

    # function to add an edge to an instance of graph
    def add_edge(self, edge):
        edge_rev = tuple([edge[1], edge[0]])

        if edge[0] not in self.VerticesSet:
            self.add_vertex(edge[0])
        if edge[1] not in self.VerticesSet:
            self.add_vertex(edge[1])

        self.AdjList[edge[0]].append(edge[1])
        self.AdjListUndir[edge[0]].append(edge[1])
        self.AdjListUndir[edge_rev[0]].append(edge_rev[1])
        self.EdgesList.append(edge)
        self.EdgesListUndir.append(edge)
        self.EdgesListUndir.append(edge_rev)

    # checks if it is possible to reach the vertex with ind target from the vertex with ind root
    def dfs(self, root, target, directed):
        if root == target:
            return True

        if directed:
            al = self.AdjList
        else:
            al = self.AdjListUndir
        visited = set()
        stack = [root]
        while stack != []:
            cur = stack.pop(-1)
            if cur == target:
                return True
            visited.add(cur)
            for adj in al[cur][::-1]:
                if adj not in visited:
                    stack.append(adj)
        return False

    # checks whether an instance of graph is connected (connected means doesn't have disjoint parts)
    # using DFS for the undirected version of the instance
    def is_connected(self,init_state):
        for v in self.VerticesList:
            if not self.dfs(states_dict[init_state], v, False):
                return False
        return True

    # checks whether all nodes are reachable from the initial one using DFS for the directed version of the instance
    def all_reachable(self, init_state):
        for v in self.VerticesList:
            if v == states_dict[init_state]:
                continue
            if not self.dfs(states_dict[init_state], v, True):
                return False
        return True


# input file parsing and checking for errors
#------------------------------------------------------------------------------------------
inp = open("input.txt", "r")
data = inp.readlines()
inp.close()
states_list, alpha, init_st_list, fin_st_list, trans = [line.split("=") for line in data]
del data

# checking the correctness of input lines' titles: throws E1
if states_list[0]!="states" or alpha[0]!="alpha" or init_st_list[0]!="initial" or fin_st_list[0]!="accepting" or trans[0]!="trans":
    print("E1: Input file is malformed")
    exit()

states_list, alpha, init_st_list, fin_st_list, trans = map(parse, [states_list, alpha, init_st_list, fin_st_list, trans])

# checking whether the number of elements in the initial states, states and alpha
# sets is appropriate: throws E1
if len(init_st_list) > 1 or states_list==[''] or alpha==['']:
    print("E1: Input file is malformed")
    exit()

states = set(states_list)
alpha = set(alpha)
init_st = set(init_st_list)
fin_st = set(fin_st_list)

# checking whether all the elements of alpha are appropriate: throws E1
for a in alpha:
    if not appropriate_name(a, False):
        print("E1: Input file is malformed")
        exit()

# enumerating the states by creating a dictionary with elements of form "state: ind", where indices: 0,1,2,3,...
states_dict = dict()
ind = 0
for item in states_list:
    # for each state check if it is named appropriately: throws E1
    if not appropriate_name(item, True):
        print("E1: Input file is malformed")
        exit()
    states_dict[item] = ind
    ind += 1
del states_list

# checking whether there is the initial state: throws E2
if len(init_st) == 0 or init_st=={''}:
    print("E2: Initial state is not defined")
    exit()

# checking whether there are final states: throws E3
if len(fin_st) == 0 or fin_st=={''}:
    print("E3: Set of accepting states is empty")
    exit()

# checking whether the initial state belongs to the set of states: throws E4
if init_st_list[0] not in states:
    print("E4: A state '"+ init_st_list[0] +"' is not in the set of states")
    exit()

# checking whether all the final states belong to the set of states: throws E4
for f in fin_st_list:
    if f not in states:
        print("E4: A state '" + f + "' is not in the set of states")
        exit()

# creating a graph based on the transitions and indices (that are stored as the dictionary values) for their states
g = Graph()
for s in states:
    g.add_vertex(states_dict[s])

# trans_objs[ind] is the list of all transitions from the state[ind] in form of Transition class objects
# trans_objs will be used for completeness and determinism checking
# checking if all states belong to the set of states: throws E4
trans_objs = [[] for _ in range(len(states))]
for t in trans:
    cur_trans_obj = Transition(t)
    if cur_trans_obj.state1 not in states:
        print("E4: A state '" + cur_trans_obj.state1 + "' is not in the set of states")
        exit()
    if cur_trans_obj.state2 not in states:
        print("E4: A state '" + cur_trans_obj.state2 + "' is not in the set of states")
        exit()
    # for each transition, checking whether its token belongs to alpha: throws E5
    if cur_trans_obj.token not in alpha:
        print("E5: A transition '" + cur_trans_obj.token + "' is not represented in the alphabet")
        exit()
    trans_objs[states_dict[cur_trans_obj.state1]].append(cur_trans_obj)
    g.add_edge(tuple([states_dict[cur_trans_obj.state1], states_dict[cur_trans_obj.state2]]))
del trans

# checking whether the graph (FSA) contains disjoint vertices (states): throws E6
if not g.is_connected(list(init_st)[0]):
    print("E6: Some states are disjoint")
    exit()

# checking completeness and determinism of the FSA: throws E7
complete = True
token_occured = dict()
for s in trans_objs:
    if len(s) > 0:
        for a in alpha:
            token_occured[a] = 0
        contains_all_tokens = True
        for a in alpha:
            for i in range(len(s)):
                if s[i].token == a:
                    token_occured[a] += 1
                    if token_occured[a] >= 2:  # nondeterministic
                        print("E7: FSA is nondeterministic")
                        exit()
            if token_occured[a]==0:
                contains_all_tokens = False
        if not contains_all_tokens:
            complete = False

del token_occured

# Applying Kleene's algorithm
#------------------------------------------------------------------------------------------
n = len(states)
# i,j = 0,..,n-1 ;  k = -1,...,n-1 => shape(r) = (n+1) x n x n
r = [[["" for _ in range(n)] for _ in range(n)] for _ in range(n+1)] # index order: k,i,j

# initial regular expressions, i.e. for k=-1
for i in range(n):
    for j in range(n):
        insertion_sort(trans_objs[i])
        for t in range(len(trans_objs[i])):
            trans = trans_objs[i][t]
            if states_dict[trans.state2]==j:
                if trans.token=="":
                    r[0][i][j] += "eps"
                else:
                    r[0][i][j] += trans.token
                if t!=len(trans_objs[i])-1:
                    r[0][i][j] += '|'
            else:
                r[0][i][j] = r[0][i][j][:-1] # remove redundant '|'
        if r[0][i][j]=="":
            r[0][i][j] = "{}"
        if i==j:
            if r[0][i][j]!="{}":
                r[0][i][j] += "|eps"
            else:
                r[0][i][j] = "eps"
        r[0][i][j] = "("+r[0][i][j]+")"


# building the rest regular expressions on top of the previously built ones
for k in range(1,n+1):
    for i in range(n):
        for j in range(n):
            r[k][i][j] = "("+r[k-1][i][k-1]+r[k-1][k-1][k-1]+"*"+r[k-1][k-1][j]+"|"+r[k-1][i][j]+")"


# uniting the regular expressions for each final state
result = ""
for f in fin_st_list:
    if r[n][0][states_dict[f]]=="":
        r[n][0][states_dict[f]] = "{}"
    result += r[n][0][states_dict[f]] + "|"

if result=="":
    result = "{}"
else:
    result = result[:-1] # remove redundant '|'

print(result)