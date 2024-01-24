import numpy as np
import random as random

def choose_action(state,row_n, col_n, policy):
    # obtain possible actions
    possible_actions= [0, 1, 2, 3]  # up, down, left, right
    epsilon=0.1 #eps-greedy
    row_num, clo_num = state  
    # choose action
    if np.random.uniform() < epsilon:
        action = np.random.choice(possible_actions)  # ε-greedy
    else:
        action = possible_actions[np.argmax(policy[row_num, clo_num, possible_actions])]  # Max Q(s, a)
    return action

    
def get_next_state(state, action, wall, row_n, col_n):
    P=0.02 # transition probabilities
    row_num, clo_num = state
    possible_of_next_state=[(row_num - 1, clo_num),(row_num + 1, clo_num), (row_num, clo_num - 1), (row_num, clo_num + 1) ]
    next_state = state
    if  action == 0:  # up
        if np.random.uniform()>P: 
           next_state = possible_of_next_state[0]
        else:
            possible_of_next_state.remove(possible_of_next_state[0])
            next_state =random.choice(possible_of_next_state)            
    elif action == 1:  # down
         if np.random.uniform()>P: 
            next_state = possible_of_next_state[1]
         else:
            possible_of_next_state.remove(possible_of_next_state[1])
            next_state =random.choice(possible_of_next_state)     
    elif action == 2:  # left
        if np.random.uniform()>P: 
           next_state = possible_of_next_state[2]
        else:
            possible_of_next_state.remove(possible_of_next_state[2])
            next_state =random.choice(possible_of_next_state)     
    elif action == 3:  # right
        if np.random.uniform()>P: 
           next_state = possible_of_next_state[3]
        else:
            possible_of_next_state.remove(possible_of_next_state[3])
            next_state =random.choice(possible_of_next_state)     
    row_num, clo_num = next_state                     
    #when encounters the wall, it would go back to the original state.
    if row_num == -1 or clo_num == -1 or row_num == row_n or clo_num == col_n or next_state in wall:
        next_state=state     
    # when encounters the wall, it would go back to the original state.      
    return next_state




def get_best_reward_route(grid, begin_cord, exit_coord, wall, max_iterations):

    action_n = 4
    row_n, col_n = grid.shape
 
    alpha = 0.3  # learning rate
    gamma = 0.95 # discount factor
    
 
    # initial action value.
    policy = np.zeros((row_n, col_n, action_n))
    best_route = []
    max_route_reward = -np.Inf
    for n_iter in range(max_iterations):
        #initial location
        n_counter=0
        state = begin_cord
        route = [state]
        while state != exit_coord:  # reaches end 
              if n_counter>1000:# max 1000 states of each eposide.
                 break
             
               # obtain an action ues greedy
              action=choose_action(state,row_n, col_n, policy)
                    
              # take an action, renew the state.
              next_state=get_next_state(state, action, wall, row_n, col_n) 
              
              # obtain instant rewards
              reward = grid[next_state]
              
              # obtain maxa Q(s,a)
              row_num, clo_num = next_state
              val=-np.Inf
              possible_actions=[0,1,2,3]
              for i in range(len(possible_actions)):
                  if policy[next_state][possible_actions[i]]> val:
                      val=policy[next_state][possible_actions[i]]
                      
              # update policy
              policy[state][action] = (1 - alpha) * policy[state][action] + alpha * (reward + gamma * val)
 
              # update state
              state = next_state
              route.append(state)
              n_counter=n_counter+1

        route_reward = sum(grid[state] for state in route)
        if max_route_reward <= route_reward:
            max_route_reward = route_reward
            best_route = route.copy()
            print(f"iteration: {n_iter}")

 
        route.clear()
 
    print('-' * 100)
    return best_route, max_route_reward,policy        

if __name__ == '__main__':
    # create the 18*18 
    grid = -np.ones((18, 18))
    # start locations
    begin_cord = (14, 3)
    # end locations(goal state)
    exit_coord = (2, 12)
 
    # the reward for the goal state 200
    grid[exit_coord] = 200

# This part is the  bump net
    bump=[(0,10),(0,11),
         (1,0),(1,1),(1,2),
         (4,0),(4,8),(4,16),
         (5,16),
         (6,1),(6,9),(6,10),(6,16),
         (7,16),
         (11,10),(11,11),
         (13,0),(13,1),
         (14,16),(14,17),
         (15,6)]
for item in bump:
    (i,j)=item
    grid[i,j]=-11
 
# This part is the oil net
oil=[(1,7),(1,15),
         (3,1),
         (4,5),
         (9,17),
         (14,9),
         (15,9),
         (16,13),(16,16),
         (17,6)]  
for item in oil:
    (i,j)=item
    grid[i,j]=-6
    
# This part is the wall net.
wall=[(1,4),
      (2,4),
      (3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(3,9),(3,10),(3,11),(3,12),(3,13),(3,14),(3,15),
      (4,2),
      (5,2),(5,5),(5,8),(5,14),
      (6,2),(6,5),(6,8),(6,11),(6,12),(6,13),(6,14),
      (7,5),(7,8),(7,14),
      (8,5),(8,8),(8,14),
      (9,0),(9,1),(9,2),(9,3),(9,5),(9,8),(9,9),(9,14),
      (10,5),(10,9),(10,12),(10,14),(10,15),(10,16),
      (11,2),(11,3),(11,4),(11,5),(11,6),(11,9),(11,12),(11,16),
      (12,6),(12,9),(12,12),(12,16),
      (13,6),(13,9),(13,12),
      (14,6),(14,12),(14,13),(14,14),(14,15),
      (16,0),(16,1),(16,6),(16,7),(16,8),(16,9),(16,10),(16,11)]
for item in wall:
    (i,j)=item
    grid[i,j]=-1    

for n_indepent in range(1):#10 indenpent runs
    print(f"The result of the {n_indepent+1}-th indenpent run(Q-learning):\n")
    max_reward_route, best_route,policy = get_best_reward_route(grid, begin_cord, exit_coord, wall, max_iterations=100000)
    print("Training is finished")



# The following part is to use our policy to find the part.
print("After training, we can use our policy to find the path,(the action that has the maximum Q value, not eps-greedy), we get:")
# use our policy to find the way:
row_n, col_n = grid.shape
best_route = []       
state = begin_cord
route = [state]
while state != exit_coord:  # reaches end 
      row_num, clo_num = state
      possible_actions=[0,1,2,3]
      action = possible_actions[np.argmax(policy[row_num, clo_num, possible_actions])]                                                   
      # take an action, renew the state.
      if  action == 0:  # up
          next_state = (row_num - 1, clo_num)    
      elif action == 1:  # down
          next_state = (row_num + 1, clo_num)
      elif action == 2:  # left
          next_state = (row_num, clo_num - 1)
      elif action == 3:  # right
          next_state = (row_num, clo_num + 1)          
      # update state
      state = next_state
      route.append(state)
              
route_reward = sum(grid[state] for state in route)
best_route = route.copy()
print(f"best_route：{best_route}\n best_route_reward：{route_reward}\n ")




















































