import numpy as np

def overlay_neurons(neuron_footprints, n1, n2, n3):
    '''Overlay the spatial filters of different neurons in RGB image channels.

    Parameters
    ----------
    neuron_footprints: numpy array, spatial filter for all neurons, size is height
                       x width x neuron number.
    n1, n2, n3: int, index of neurons to be shown. To create different color combinations
                they can also be passed twice.
    ---------------------------------------------------------------------------
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    all_cells = Image.fromarray(np.uint8(np.stack((np.sum(neuron_footprints,axis=2),np.sum(neuron_footprints,axis=2),
                                    np.sum(neuron_footprints,axis=2)),axis=2) * 255), mode = 'RGB') #In RGB mode PIL needs a uint8 image with dimensions width x height x RGB
    examples = Image.fromarray(np.uint8(np.stack((neuron_footprints[:,:,n1] / np.max(neuron_footprints[:,:,n1]),
                     neuron_footprints[:,:,n2] / np.max(neuron_footprints[:,:,n2]),
                     neuron_footprints[:,:,n3] / np.max(neuron_footprints[:,:,n3])), axis=2) * 255), mode='RGB')
    composite = np.array(Image.blend(all_cells, examples, 0.5))
    plt.figure()
    plt.imshow(composite)
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


def create_subplot_axes(n_rows, n_cols, fig_size = [12, 12]):
    '''Create a figure with a specified subplot grid and return axes objects
    as a list.

    Parameters
    ---------
    n_rows: int, number of rows on the plot
    n_cols: int, number of columns on the plot
    fig_size: list, width and height of the figure

    Returns
    -------
    axes_list: list, axes objects for all the subplot. The order is first along
            the columns and then rows
    ---------------------------------------------------------------------------
    '''
    import matplotlib.pyplot as plt

    subplot_num = n_rows * n_cols
    axes_list = []
    fig = plt.figure(figsize = fig_size)
    for k in range(subplot_num):
        axes_list.append(fig.add_subplot(n_rows, n_cols, k+1)) #Here first element has to be 1

    return axes_list

def plot_task_var_betas(task_var_betas, time_betas, task_var_name, neuron_id, aligned_segment_start, ax, colors=['#062999','#b20707'], spacer=7, var_value_strings=['Left', 'Right'], frame_rate=30):
    
    beta_traces = np.stack((time_betas[:,neuron_id], task_var_betas[:,neuron_id]),axis=1)
    alignment_position_x_vect = [-1, 0, 6, 0] #These are the indices of the x_vect element where the alignment happend. 
    #Hard coded here, for simplicity.
    
    for n in range(beta_traces.shape[1]): #Loop through the average traces
        for k in range(len(aligned_segment_start)): #Loop through the different task segments
            
            if k < len(aligned_segment_start) - 1 : #Split by different segments
                plot_idx = np.arange(aligned_segment_start[k], aligned_segment_start[k+1]) #Extract indices of the current segment
                x_vect = (plot_idx + k*spacer) / frame_rate #Create a time vector to plot the data to
                line_label = None 
            else:
                plot_idx = np.arange(aligned_segment_start[k], beta_traces.shape[0])
                x_vect = (plot_idx + k*spacer) /frame_rate
                line_label = var_value_strings[n] #only use a label on the last segment
            ax.plot(x_vect, beta_traces[plot_idx,n], color = colors[n], label = line_label)
            
            if n == 0: #Plot vertical alignment timepoints only once in the loop
                ax.axvline(x_vect[alignment_position_x_vect[k]], color = 'k', linewidth = 0.8, linestyle = '--')
            
    
    #Do some figure formating
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Regression weight')
    ax.set_title(f'{task_var_name} for neuron {neuron_id}')
    ax.legend()
    ax.set_ylim([-2, 2.2]) #Pre-set for this data
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_bounds([ax.get_yticks()[0], ax.get_yticks()[-2]]) 
    ax.spines['bottom'].set_bounds([ax.get_xticks()[1], ax.get_xticks()[-2]]) 
    return

    
def plot_mean_trace(Y_3d,
                    task_var, #the binary array of task variables for each trial (stimulus, choice, etc.)
                    task_var_name, #used for figure title
                    neuron_id, 
                    aligned_segment_start, 
                    ax,
                    colors=['#062999','#b20707'], 
                    spacer=7, 
                    var_value_strings=['Left', 'Right'],
                    frame_rate=30):

    alignment_position_x_vect = [-1, 0, 6, 0] #These are the indices of the x_vect element where the alignment happend. 
    
    average_trace = np.stack((np.mean(Y_3d[:, neuron_id, task_var==0], axis=1),
                              np.mean(Y_3d[:, neuron_id, task_var==1], axis=1))).T # stack the different conditions along a new dimension
    
    sem_trace = np.stack((np.std(Y_3d[:, neuron_id, task_var == 0],axis=1) / np.sqrt(np.sum(task_var == 0)),
                          np.std(Y_3d[:, neuron_id, task_var == 1],axis=1) / np.sqrt(np.sum(task_var == 1)))).T
    
    for n in range(average_trace.shape[1]): #Loop through the conditions (left and right in the case of choice)
        for k in range(len(aligned_segment_start)): #Loop through the different task segments
            
            if k < len(aligned_segment_start) - 1 : #Split by different segments
                plot_idx = np.arange(aligned_segment_start[k], aligned_segment_start[k+1]) #Extract indices of the current segment
                x_vect = (plot_idx + k*spacer) / frame_rate #Create a time vector to plot the data to
                line_label = None 
            else:
                plot_idx = np.arange(aligned_segment_start[k], Y_3d.shape[0])
                x_vect = (plot_idx + k*spacer) /frame_rate
                line_label = var_value_strings[n] #only use a label on the last segment

            ax.fill_between(x_vect, average_trace[plot_idx,n] - sem_trace[plot_idx,n], average_trace[plot_idx,n] + sem_trace[plot_idx,n], color=colors[n], alpha=0.5, linewidth=0)
            ax.plot(x_vect, average_trace[plot_idx,n], color=colors[n], label=line_label)

            if n == 0: #Plot vertical alignment timepoints only once in the loop
                ax.axvline(x_vect[alignment_position_x_vect[k]], color = 'k', linewidth = 0.8, linestyle = '--')
            
    #Do some figure formating
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Fluorescence (A.U.)')
    ax.set_title(f'{task_var_name} for neuron {neuron_id}')
    ax.legend()
    ax.set_ylim([-5, 40]) #Pre-set for this data
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_bounds([ax.get_yticks()[0], ax.get_yticks()[-1]]) 
    ax.spines['bottom'].set_bounds([ax.get_xticks()[1], ax.get_xticks()[-2]]) #In this case the plotting is bounded by invisible ticks, thus 1 and -2
    return