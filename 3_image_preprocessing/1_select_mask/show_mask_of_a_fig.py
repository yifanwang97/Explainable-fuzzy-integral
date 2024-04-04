import numpy as np
import matplotlib.pyplot as plt
import copy


masks = np.load('example_folder/example.npy').astype("int")


for j in range(1, masks.max() + 1):
    mask = copy.deepcopy(masks)
    mask[mask != j] = 0
    mask[mask == j] = 1
    plt.imshow(mask, cmap='Greys_r')
    plt.axis('off')
    plt.show()
    fig_name = 'mask/' + 'mask' + str(j) + '.jpg'
    plt.savefig(fig_name, bbox_inches='tight', pad_inches = -0.1)
    print(j)



