import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from task_utils import state2idcs

def plot_1D_arrows(Pol,maze,term_states_for_plotting,
                    question=True,box=False,box_size_adjustement=False,box_ec='k',jointbox=False,
                  ms=15,
                  asym_color=False):

    arrow_markers = np.array(['$\u2190$','$\u2192$'])
    if asym_color:
        color_set = np.array([sns.color_palette()[3],'k'])
    else:
        color_set = np.array(['k','k'])

    # loop over states
    for s in range(Pol.shape[0]):
        if s not in term_states_for_plotting:

            # policy for single state
            s_idcs = state2idcs(s,maze,order='C')
            pi = Pol[s,:]

            markers = arrow_markers[np.where(pi==np.max(pi))[0]] # allows for ties.
            mults=np.ones(len(markers))
            colors = color_set[np.where(pi==np.max(pi))[0]]

            # multiple directions
            if len(markers)>1:

                for marker,mult,color in zip(markers,mults,colors):
                    yoffset=0
                    xoffset=0

                    if marker==arrow_markers[0]:
                        xoffset = -0.18#+0.07
                    elif marker==arrow_markers[1]:
                        xoffset= 0.18#-0.07

                    plt.plot(s_idcs[1]+xoffset,s_idcs[0]+yoffset,marker=marker,
                        color=color,linewidth=0.25,linestyle='--',ms=ms*mult)

            # single direction
            else:
                marker = markers[0]
                color = colors[0]
                plt.plot(s_idcs[1],s_idcs[0],marker=marker,color=color,linewidth=0.25,linestyle='--',ms=ms)

        else:
            pass


def plot_policy(task,V,Pol,core_states,Alpha=None,asym_color=False,ms=35,alpha_fontsize=8):

    maze = task.maze

    Qrange = [-15,15]
    n_colors=100
    cm_Q = sns.light_palette("red",int(n_colors/2))[::-1]+\
            [(0.96, 0.96, 0.96)]+sns.light_palette("green",int(n_colors/2))

    # plot value
    fig,ax = plt.subplots(1,1,figsize=(6,4),dpi=200)
    im_value = ax.imshow(np.zeros_like(V.reshape(maze.shape)),
                                      interpolation='none',origin='upper',
                                      cmap = matplotlib.colors.ListedColormap(cm_Q),
                                      vmax=Qrange[1],
                                      vmin=Qrange[0],
                                      )

    # add arrows
    plot_1D_arrows(Pol,maze,[],
                   question=True,box=False,box_size_adjustement=False,
                   box_ec='k',jointbox=False,
                   ms=ms,asym_color=asym_color)

    # add alphas
    if Alpha is not None:
        for si,s in enumerate(core_states):
            idcs = state2idcs(s,maze,order='C')
            adjusted_alpha =np.round(Alpha[si][0],2)

            if alpha_fontsize==8:
                if si==0:
                    text=r'$\alpha$='+str(adjusted_alpha)
                    offset = np.array([0,-0.35])
                else:
                    text=adjusted_alpha
                    offset = np.array([0.2,-0.35])
            else:
                if si==0:
                    text=r'$\alpha$='+str(adjusted_alpha)
                    offset = np.array([-0.15,-0.35])
                else:
                    text=adjusted_alpha
                    offset = np.array([0.1,-0.35])
            plt.annotate(text,np.array(idcs)[::-1]+offset,fontsize=alpha_fontsize,ha='left')

    # fix grid
    ax.set_yticks(np.arange(0, maze.shape[0], 1));
    ax.set_yticks(np.arange(-.5, maze.shape[0], 1), minor=True);
    ax.grid(True,which='minor', color='k', linestyle='-', linewidth=0.5,axis='both')
    ax.set_xticks(np.arange(0, maze.shape[1], 1));
    ax.set_xticks(np.arange(-.5, maze.shape[1], 1), minor=True);
    ax.grid(True,which='minor', color='k', linestyle='-', linewidth=0.5,axis='both')
    plt.xticks([])
    plt.yticks([])
    outer_lw=3
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth(str(outer_lw))

    # add start
    plt.annotate('start',[0,0.34],fontsize=12,ha='center')
