import numpy as np
import matplotlib.pyplot as plt
import cv2

if __name__=='__main__':
    fig = plt.figure(figsize=(3, 2))
    columns = 4
    rows = 2

    ax = []
    img_list = ['event_conv1d.png', 'event_crnn_1d.png', 'event_conv2d.png', 'event_crnn_2d.png', 'voice_conv1d.png', 'voice_crnn_1d.png', 'voice_conv2d.png', 'voice_crnn_2d.png']
    for i in range(columns*rows):
        img = cv2.imread(img_list[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # create subplot and append to ax
        ax.append( fig.add_subplot(rows, columns, i+1) )
        ax[-1].set_title("ax:"+str(i), fontsize=4)  # set title
        ax[-1].grid(False)
        ax[-1].axis('off')
        plt.imshow(img)

    plt.savefig('imp.png', dpi=1000)