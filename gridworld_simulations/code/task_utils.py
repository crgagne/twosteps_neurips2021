import numpy as np

def action_str_to_num(a):
    if a == 'up':
        return(0)
    elif a == 'down':
        return(1)
    elif a == 'right':
        return(2)
    elif a == 'left':
        return(3)

def idcs2state(idcs: list, maze, order='F'):
    '''Convert state idcs to state id.

    Notes:
       order='C' is row major [[1,2,3],[4,5,6]]
       order='F' is column major [[1,3,4],[2,4,6]]

    Example:
        maze = np.zeros((3,4))
        for s in range(Ns_wo_absorb):
            idx = state2idcs(s,maze,order='F')
            s_rec = idcs2state(idx,maze,order='F')
            maze[idx[0],idx[1]]=s_rec
        print(maze)
    '''
    si = idcs[0]
    sj = idcs[1]
    side_j = maze.shape[1]
    side_i = maze.shape[0]

    if order =='C':
        return si*side_j + sj
    if order =='F':
        return sj*side_i + si

def state2idcs(s: int, maze, state_mult=1,order='F'):
    '''Convert state id to state idcs. Inverts previous function.
    '''
    # convert state to location id
    num_locs = maze.shape[0]*maze.shape[1]
    loc = s%(num_locs)

    # convert location id to i,j coordinates
    if order =='C':
        side_j = maze.shape[1]
        si = loc // side_j
        sj = loc % side_j
    elif order=='F':
        side_j = maze.shape[0] # swapped
        sj = loc // side_j # swapped
        si = loc % side_j

    return [int(si), int(sj)]
