def get_cmap_neurons():
    import numpy as np

    return np.array([[218,166,156], [164,30,79], [119,161,187],[92,114,183]])/255

def assign_color(net, gid):
    import numpy as np
    import matplotlib

    cm_neurons = get_cmap_neurons()
    cmap = matplotlib.colors.ListedColormap(cm_neurons)

    cell_type_colors = {'L2_basket': cmap.colors[0], 'L2_pyramidal': cmap.colors[1], 
                'L5_basket': cmap.colors[2], 'L5_pyramidal': cmap.colors[3]}

    for cell_type, gid_range in net.gid_ranges.items():
        if cell_type in list(cell_type_colors.keys()):
            if gid in gid_range:
                return cell_type_colors[cell_type]