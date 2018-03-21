import matplotlib.pyplot as plt
import csv
from matplotlib.widgets import Button
from matplotlib.text import Annotation
import seaborn as sns

DATA_LABELS = False # Boolean to control Data label callouts in points

# Config for plotting groups
X_THRESHOLD = 0.5 # KL_DIVERGENCE
Y_THRESHOLD = 0.5 # LEX_DISTANCE

def assign_group(x,y):
    QUADRANT_COLORS = {1:"#e41a1c", 2:"#377eb8", 3:"#4daf4a", 4:"#BDB76B"}
    
    if x >= X_THRESHOLD and y >= Y_THRESHOLD:
        c = QUADRANT_COLORS[1]
    elif x < X_THRESHOLD and y >= Y_THRESHOLD:
        c = QUADRANT_COLORS[2]
    elif x < X_THRESHOLD and y < Y_THRESHOLD:
        c = QUADRANT_COLORS[3]
    else:
        c = QUADRANT_COLORS[4]
    return c

if __name__ == "__main__":
    from sys import argv
    sns.set()
    # print(argv[1])
    x, y, c, generated_labels = [], [], [], []
    with open(argv[1],'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        next(plots, None)  # skip the headers
        for row in plots:
            px, py = float(row[6]), float(row[7])
            d1 = row[0].split("\\")[-1].split(".")[0]
            d2 = row[3].split("\\")[-1].split(".")[0]
            lbl = "{0}.{1}|{2}.{3}".format(d1, row[1], d2, row[4])
            # print("Read {0} and {1}".format(px, py))
            x.append(px)
            y.append(py)
            c.append(assign_group(px, py))
            generated_labels.append(lbl)
        
        # fig = plt.figure(figsize=(20, 16))
        fig = plt.figure()
        ax = plt.subplot()
        def draw_scatterplot():
            ax.scatter(x, y, c=c, picker=True, s=10)
            ax.set_xlabel('KL Divergence')
            ax.set_ylabel('Lexicographical Distance')
            ax.set_title('KL vs LD\n{0}'.format(argv[1]))

        draw_scatterplot()

        def annotate(axis, text, x, y):
            text_annotation = Annotation(text, xy=(x, y), xycoords='data', size=9)
            axis.add_artist(text_annotation)
        
        def onpick(event):
            # step 1: take the index of the dot which was picked
            ind = event.ind

            # step 2: save the actual coordinates of the click, so we can position the text label properly
            label_pos_x = event.mouseevent.xdata
            label_pos_y = event.mouseevent.ydata

            # just in case two dots are very close, this offset will help the labels not appear one on top of each other
            offset = 0

            # if the dots are to close one to another, a list of dots clicked is returned by the matplotlib library
            for i in ind:
                # step 3: take the label for the corresponding instance of the data
                label = generated_labels[i]

                # step 4: log it for debugging purposes
                print("index", i, label)

                # step 5: create and add the text annotation to the scatterplot
                if DATA_LABELS:
                    annotate(ax,label,label_pos_x + offset,label_pos_y + offset)

                # step 6: force re-draw
                ax.figure.canvas.draw_idle()

                # alter the offset just in case there are more than one dots affected by the click
                if DATA_LABELS:
                    offset += 0.1

        # connect the click handler function to the scatterplot
        fig.canvas.mpl_connect('pick_event', onpick)

        # create the "clear all" button, and place it somewhere on the screen
        ax_clear_all = plt.axes([0.0, 0.0, 0.1, 0.05])
        button_clear_all = Button(ax_clear_all, 'Reset')

        def onclick(event): # On click of Button
            # step 1: we clear all artist object of the scatter plot
            ax.cla()

            # step 2: we re-populate the scatterplot only with the dots not the labels
            draw_scatterplot()

            # step 3: we force re-draw
            ax.figure.canvas.draw_idle()
        # link the event handler function to the click event on the button
        button_clear_all.on_clicked(onclick)

        # initial drawing of the scatterplot
        plt.plot()
        print("scatterplot done")

        # present the scatterplot
        plt.show()
        # plt.scatter(x,y)
        # plt.axhline(y=Y_THRESHOLD)
        # plt.axvline(x=X_THRESHOLD)

        # plt.show()