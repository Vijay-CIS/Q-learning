import numpy as np
import pylab as pl
import networkx as nx
cities = {0:'Kolkata',1:'Delhi',2:'Mumbai',3:'Bangalore',4:'Hyderabad',
          5:'Ahmedabad',6:'Pune',7:'Visakapatnam',8:'Patna',9:'Jaipur'
          ,10:'Chennai'}
edges = [(3,10),(4,10),(7,10),(8,0),(9,1),(3,2),(4,2),
         (5,2),(6,2),(4,3),(6,3),(6,4),(7,4),(6,5),(9,5)
         ,(8,1),(0,1),(7,0)]
print(cities)
dist = {'010':1661,'110':2208,'210':1336,'310':347,'410':627,'510':1849,'610':1192,'710':793,
        '810':2097,'910':2066,'10':1561,'20':1934,'30':1870,'40':1480,'50':1966,'60':1822,
        '70':883,'80':549,'90':1577 ,'21':1423,'31':2163,'41':1572,'51':948,'61':1472,'71':1785,
        '81':1058,'91':273,'32':983,'42':708 ,'52':524,'62':148,'72':1335,'82':1723,'92':1145,
        '43':578,'53':1498,'63':841,'73':1001,'83':2062 ,'93':1929,'54':1176,'64':561,'74':617,
        '84':1474,'94':1462,'65':661,'75':1803,'85':1771,'95':677 ,'76':1183,'86':1679,'96':1283,
        '87':1220,'97':1788,'89':1076}

goal = 10
G = nx.Graph()
G.add_edges_from(edges)
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos)
pl.show()
MATRIX_SIZE = 11
M = np.matrix(np.ones(shape =(MATRIX_SIZE, MATRIX_SIZE)))
M *= -1

for point in edges:
	if point[1] == goal:
		M[point] = 100
	else:
		M[point] = 0

	if point[0] == goal:
		M[point[::-1]] = 100
	else:
		M[point[::-1]]= 0
		# reverse of point

M[goal, goal]= 100
print(M)
# add goal point round trip
Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))

gamma = 0.75
# learning parameter
initial_state = 1

# Determines the available actions for a given state
def available_actions(state):
	current_state_row = M[state, ]
	available_action = np.where(current_state_row >= 0)[1]
	return available_action

available_action = available_actions(initial_state)

# Chooses one of the available actions at random
def sample_next_action(available_actions_range):
	next_action = int(np.random.choice(available_action, 1))
	return next_action


action = sample_next_action(available_action)

def update(current_state, action, gamma):

    max_index = np.where(Q[action, ] == np.max(Q[action, ]))[1]
    if max_index.shape[0] > 1:
    	max_index = int(np.random.choice(max_index, size = 1))
    else:
    	max_index = int(max_index)
    max_value = Q[action, max_index]
    Q[current_state, action] = M[current_state, action] + gamma * max_value
    if (np.max(Q) > 0):
    	return(np.sum(Q / np.max(Q)*100))
    else:
    	return (0)
# Updates the Q-Matrix according to the path chosen

update(initial_state, action, gamma)
scores = []
for i in range(1000):
	current_state = np.random.randint(0, int(Q.shape[0]))
	available_action = available_actions(current_state)
	action = sample_next_action(available_action)
	score = update(current_state, action, gamma)
	scores.append(score)

# print("Trained Q matrix:")
# print(Q / np.max(Q)*100)
# You can uncomment the above two lines to view the trained Q matrix

# Testing
current_state = 7
current_state1 = current_state
steps = [current_state]

while current_state != 10:

	next_step_index = np.where(Q[current_state, ] == np.max(Q[current_state, ]))[1]
	if next_step_index.shape[0] > 1:
		next_step_index = int(np.random.choice(next_step_index, size = 1))
	else:
		next_step_index = int(next_step_index)
	steps.append(next_step_index)
	current_state = next_step_index

print("Most efficient path:")
n=0
for i in steps:
    n=n+1
    if n == len(steps):
        print(cities[i])
        break
    print(cities[i],"->",end=" ")

print('Distance:')
str1=str(current_state1)
str2=str(goal)
str3=str1+str2
str4=str2+str1
if str3 in dist.keys():
    print(dist[str3])
else:
    print(dist[str4])

# pl.plot(scores)
# pl.xlabel('No of iterations')
# pl.ylabel('Reward gained')
# pl.show()
