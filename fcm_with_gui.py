from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import os
import numpy as np
import skfuzzy as fuzz
from scipy.spatial.distance import cdist
from scipy import optimize
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils import *


class gui:
    def __init__(self, master):
        # Create a container
        frame = Frame(master)
        # Create 2 button
        Label(frame, text = "Number of Clusters:").grid(row=0, column = 0)
        Label(frame, text = "Number of Data Points:").grid(row=1, column = 0)
        Label(frame, text = "Beta Value:").grid(row=0, column = 3)
        Label(frame, text = "Best Beta Value Precision:").grid(row=1, column = 3)
        self.num_centers = Entry(frame)
        self.num_data = Entry(frame)
        self.beta = Entry(frame)
        self.step_size = Entry(frame)
        self.num_centers.grid(row=0, column=1)
        self.num_data.grid(row=1, column=1)
        self.beta.grid(row=0, column=4)
        self.step_size.grid(row = 1, column =4)
        Button(frame, text='Generate new data', command = self.gen_data_button).grid(row=3, column=0, sticky=W, pady=4)
        Button(frame, text='Result of Chebyshev distance', command = self.show_answer_chebyshev).grid(row=3, column=1, sticky=W, pady=4)
        Button(frame, text='Result of Weight Euclidean Distance', command = self.show_answer_weight_euclidean).grid(row=3, column=2, sticky=W, pady=4)
        # Button(frame, text='Go to Result', command = self.decrease).grid(row=3, column=1, sticky=W, pady=4)
        self.screen = Canvas(frame, width=200, height=50, background="white",highlightthickness=0)
        self.screen.create_text(50,0,anchor = "nw",fill = "NavajoWhite2" ,font = ("Helvetica",15),text = "FCM Clustering")
        self.screen.grid(row=4, column=0)


        self.fig, self.ax_data = plt.subplots()
        self.ax_data.axis('off')
        self.width, self.height = self.fig.get_size_inches()
        self.ax_chebyshev = []
        self.ax_euclidean = []

        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        # self.canvas.show()
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
        frame.pack()


    def gen_data_button(self):
        self.num_cntr = int(self.num_centers.get())
        self.data_points = int(self.num_data.get())
        # self.data_points  =200
        # self.num_cntr = 4
        # power = 0
        self.xpts, self.ypts, self.labels, self.colors= data_generator(self.num_cntr, self.data_points)
        # self.fig.clear()
        try:
            self.ax_data.clear()
            self.ax_data.remove()
            self.colorbar.remove()
            self.scatter.remove()
            # self.canvas.delete("all")
        except:
            pass

        try:
            for item in self.ax_chebyshev:
                try:
                    item.remove()
                except:
                    pass
        except:
            pass

        try:
            for item in self.ax_euclidean:
                try:
                    item.remove()
                except:
                    pass
        except:
            pass

        self.fig.set_size_inches(self.width, self.height, forward=True)
        self.ax_data = self.fig.subplots()

        for label in range(self.num_cntr):
            self.ax_data.plot(self.xpts[self.labels == label], self.ypts[self.labels == label], '.')
        t1= self.data_points * self.num_cntr
        print('new data is generated')
        self.ax_data.set_title('data points = {1:.0f};  centers = {0}'.format(self.num_cntr, t1))
        self.canvas.draw()

    def show_answer_weight_euclidean(self):
        power = float(self.beta.get())
        step_size = float(self.step_size.get())

        try:
            self.ax_data.clear()
            self.ax_data.remove()
            self.colorbar.clear()
            self.colorbar.remove()
            self.scatter.clear()
            self.scatter.remove()
            # self.canvas.delete("all")
        except:
            pass

        try:
            for item in self.ax_chebyshev:
                try:
                    item.clear()
                    item.remove()
                except:
                    pass
        except:
            pass

        try:
            for item in self.ax_euclidean:
                try:
                    item.clear()
                    item.remove()
                except:
                    pass
        except:
            pass

        self.cluster_plot_normal_data_weight_euclidean(self.xpts, self.ypts, self.colors, power, self.num_cntr, step_size)


    def show_answer_chebyshev(self):
        power = float(self.beta.get())
        step_size = float(self.step_size.get())
        # print(self.num_cntr, self.data_points, power)

        # data_points =200
        # num_cntr = 4
        # power = 0
        # if(gen == 0):
        #     xpts, ypts, label, colors= data_generator(num_cntr, data_points)
        # self.fig.clear()
        try:
            self.ax_data.remove()
        except:
            pass

        try:
            for item in self.ax_chebyshev:
                try:
                    item.clear()
                    item.remove()
                except:
                    pass
        except:
            pass

        try:
            for item in self.ax_euclidean:
                try:
                    item.clear()
                    item.remove()
                except:
                    pass
        except:
            pass

        self.cluster_plot_normal_data_chebyshev(self.xpts, self.ypts, self.colors, power, self.num_cntr, step_size)
        # print('going to frame 2....')

    def cluster_plot_normal_data_weight_euclidean(self, xpts, ypts, colors, power, ncenters, step_size):
        xpts_nor = normalization(xpts)
        ypts_nor = normalization(ypts)

        var = np.ndarray(2)
        # print(xpts.size, xpts)

        #     print('data is ',data[0]
        var.itemset(0, np.var(xpts_nor))
        var.itemset(1, np.var(ypts_nor))

    #     print('the variance is ', var)
         # Set up the loop and plot
        self.fig.set_size_inches(12, 6)
        ax = self.fig.subplots(1,2)
        # self.fig, ax = plt.subplots(1, 3, figsize=(18, 6))

        alldata = np.vstack((xpts_nor, ypts_nor))
    #     print(alldata)
        fpcs = []
        radius_lst = np.zeros(1)
        insert_index = 0
        for image_index, ax in enumerate(ax.reshape(-1), 2):
            if(image_index == 2):
                #     print(ncenters)
                cntr, u, u0, d, jm, p, fpc = cmeans(alldata, ncenters, 2, error=0.000005, maxiter=1000, var=var, method =1, init=None)

                # Store fpc values for later
                fpcs.append(fpc)

                # Plot assigned clusters, for each data point in training set
                cluster_membership = np.argmax(u, axis=0)
                x_lst = np.zeros(1)
                y_lst = np.zeros(1)
                radius_map = np.zeros(1)
                radius_lst, cov_sp_lst = generate_new_coverage_new(u, d, power, ncenters, 2)
                print('ro lst is ', radius_lst)
                for j in range(ncenters):
                    ax.plot(xpts_nor[cluster_membership == j],
                        ypts_nor[cluster_membership == j],
                        '.')

        #             #create  a circle centered at calculated ventroid with radius of the longest
        #             normal_data = normalization(d[j])
        #             # print('normal_data is ', normal_data)
        #             coverage = np.zeros(1)
        #             # coverage1 = np.zeros(1)
        #
        #
                    base_coverage = float(find_base_coverage(power))
                    max_cov = float(base_coverage * (1-base_coverage)**power)
        #             # print('max cov is ', max_cov, type(max_cov))
                    x_lst = np.hstack((x_lst, cntr[j][0]))
                    y_lst = np.hstack((y_lst, cntr[j][1]))
                    radius_map  = np.hstack((radius_map, radius_lst[j]))
        #             radius_1 = np.zeros(1)
        #             for i in range(d[j].size):
        #                 if(normal_data[i] < base_coverage):
        #                     coverage = np.hstack((coverage, u[j][i]))
        #                     # coverage1 = np.hstack((coverage1, normal_data[i]))
        #             coverage = np.delete(coverage, 0)
        #
        # #             coverage1 = np.delete(coverage1, 0)
        #             radius_1 = np.sum(coverage) / u[j].size
        #             radius_lst = np.hstack((radius_lst, radius_1))
        #             # print('base is ', base_coverage, type(base_coverage), 'radius is ', radius_1)

                    circle = plt.Circle((cntr[j][0], cntr[j][1]),
                            radius_lst[j], facecolor = cm.binary(float(cov_sp_lst[j])/max_cov), edgecolor= 'k',fill=True)

                    # circle = plt.Circle((cntr[j][0], cntr[j][1]),
                    #         radius_1,
                    #        facecolor='None', edgecolor= 'k')

                    ax.add_artist(circle)
                    ax.plot(cntr[j][0], cntr[j][1], 'rs')

                    #     membership_graph(ncenters, u, xpts, ypts)
                    ax.set_title('Beta={0}'.format(power))

                    # ax.axis('off')
                    self.ax_euclidean.insert(insert_index, ax)
                    insert_index+=1
                    #             ax.axis('off')


            elif(image_index == 3):
                radius_map = np.delete(radius_map, 0)
    #             ax.plot(range(100))
                power, power_lst, intersect_num_lst = fin_best_beta(ncenters, cntr, u, d, power, radius_lst, step_size, var)
                # Plot assigned clusters, for each data point in training set
                cluster_membership = np.argmax(u, axis=0)
                radius_lst, cov_sp_lst = generate_new_coverage_new(u, d, power, ncenters, 2)
                x_lst = np.zeros(1)
                y_lst = np.zeros(1)
                radius_map = np.zeros(1)
                for j in range(ncenters):
                    ax.plot(xpts_nor[cluster_membership == j],
                        ypts_nor[cluster_membership == j],
                        '.')

        #             #create  a circle centered at calculated ventroid with radius of the longest
        #             normal_data = normalization(d[j])
        #             # print('normal_data is ', normal_data)
        #             coverage = np.zeros(1)
        #             coverage1 = np.zeros(1)
        #
        #
                    base_coverage = float(find_base_coverage(power))
                    max_cov = float(base_coverage * (1-base_coverage)**power)
                    x_lst = np.hstack((x_lst, cntr[j][0]))
                    y_lst = np.hstack((y_lst, cntr[j][1]))
                    radius_map  = np.hstack((radius_map, radius_lst[j]))
        #             radius_1 = np.zeros(1)
        #             for i in range(d[j].size):
        #                 if(normal_data[i] < base_coverage):
        #                     coverage = np.hstack((coverage, u[j][i]))
        #                     coverage1 = np.hstack((coverage1, normal_data[i]))
        #             coverage = np.delete(coverage, 0)
        #
        # #             coverage1 = np.delete(coverage1, 0)
        #             radius_1 = np.sum(coverage) / u[j].size
        #             radius_lst = np.hstack((radius_lst, radius_1))

                    # circle = plt.Circle((cntr[j][0], cntr[j][1]),
                    #         radius_1,
                    #        color = cm.binary(radius_1/max_cov))
                    # print('base is ', base_coverage, type(base_coverage), 'radius is ', radius_1)


                    circle = plt.Circle((cntr[j][0], cntr[j][1]),
                            radius_lst[j], facecolor = cm.binary(float(cov_sp_lst[j])/max_cov), edgecolor= 'k',fill=True)


                    #
                    # # ax.add_patch(circle)
                    ax.add_artist(circle)
                    ax.plot(cntr[j][0], cntr[j][1], 'rs')

                    #     membership_graph(ncenters, u, xpts, ypts)

                    #         ax.set_title('Beta={0};Coverage={1:.4f}'.format(power, radius_1))
                    ax.set_title('Best Beta={:10.2f}'.format(power))
                    self.ax_euclidean.insert(insert_index, ax)
                    insert_index+=1
            # elif(image_index == 4):
            #     ax.axis('off')
                # ax('off')
                # ax.plot(power_lst, intersect_num_lst)
                # ax.set_title('interctect num as func of beta')
                # self.ax_euclidean.insert(insert_index, ax)
                # insert_index+=1

        self.fig.tight_layout()
        radius_map = np.delete(radius_map, 0)
        x_lst = np.delete(x_lst, 0)
        y_lst = np.delete(y_lst, 0)
        c = radius_map/np.amax(radius_map)
        self.scatter = plt.scatter(x_lst, y_lst, s=0, c=c, cmap='binary')
        # plt.grid()
        self.colorbar = plt.colorbar()  # this works because of the scatter
        # plt.show()
        self.canvas.draw()
    # plt.show()

    def cluster_plot_normal_data_chebyshev(self, xpts, ypts, colors, power, ncenters, step_size):
        xpts_nor = normalization(xpts)
        ypts_nor = normalization(ypts)

        var = np.ndarray(2)
        # print(xpts.size, xpts)

        #     print('data is ',data[0]
        var.itemset(0, np.var(xpts_nor))
        var.itemset(1, np.var(ypts_nor))

    #     print('the variance is ', var)
         # Set up the loop and plot
        self.fig.set_size_inches(12, 4, forward=True)
        ax = self.fig.subplots(1,3)

        alldata = np.vstack((xpts_nor, ypts_nor))
    #     print(alldata)
        fpcs = []
        radius_lst = np.zeros(1)
        insert_index = 0
        for image_index, ax in enumerate(ax.reshape(-1), 2):
            if(image_index == 2):
                #     print(ncenters)
                cntr, u, u0, d, jm, p, fpc = cmeans(alldata, ncenters, 2, error=0.000005, maxiter=1000, var=var, method = 2, init=None)

                # Store fpc values for later
                fpcs.append(fpc)

                # Plot assigned clusters, for each data point in training set
                cluster_membership = np.argmax(u, axis=0)
                base_coverage = find_base_coverage(power )
                for j in range(ncenters):
                    ax.plot(xpts_nor[cluster_membership == j],
                        ypts_nor[cluster_membership == j],
                        '.', color=colors[j])

                    #create  a circle centered at calculated ventroid with radius of the longest
                    normal_data = normalization(d[j])
                    # print('normal_data is ', normal_data)
                    coverage = np.zeros(1)
                    coverage1 = np.zeros(1)




                    radius_1 = np.zeros(1)
                    for i in range(d[j].size):
                        if(normal_data[i] < base_coverage):
                            coverage = np.hstack((coverage, u[j][i]))
                            coverage1 = np.hstack((coverage1, normal_data[i]))
                    coverage = np.delete(coverage, 0)

        #             coverage1 = np.delete(coverage1, 0)
                    radius_1 = np.sum(coverage) / u[j].size
                    radius_lst = np.hstack((radius_lst, radius_1))

                    # circle = plt.Circle((cntr[j][0], cntr[j][1]),
                    #             radius_1,
                    #            facecolor='None', edgecolor= 'k')

                    rectangle = plt.Rectangle((cntr[j][0]-(radius_1/2),cntr[j][1]-(radius_1/2)),
                                              radius_1,radius_1,linewidth=1,edgecolor='k',facecolor='none')

                    # ax.add_patch(circle)
                    ax.add_patch(rectangle)
                    ax.plot(cntr[j][0], cntr[j][1], 'rs')

                    #     membership_graph(ncenters, u, xpts, ypts)

                    #         ax.set_title('Beta={0};Coverage={1:.4f}'.format(power, radius_1))
                    ax.set_title('Beta={0}'.format(power))

                    # ax.axis('off')
                    self.ax_chebyshev.insert(insert_index, ax)
                    insert_index+=1
                    #             ax.axis('off')
            elif(image_index == 3):
                radius_lst = np.delete(radius_lst, 0)
    #             ax.plot(range(100))
                power, power_lst, intersect_num_lst = find_best_beta_chebyshev(ncenters, cntr, u, d, power, radius_lst, step_size)
                # Plot assigned clusters, for each data point in training set
                cluster_membership = np.argmax(u, axis=0)
                for j in range(ncenters):
                    ax.plot(xpts_nor[cluster_membership == j],
                        ypts_nor[cluster_membership == j],
                        '.', color=colors[j])

                    #create  a circle centered at calculated ventroid with radius of the longest
                    normal_data = normalization(d[j])
                    # print('normal_data is ', normal_data)
                    coverage = np.zeros(1)
                    coverage1 = np.zeros(1)


                    base_coverage = find_base_coverage(power )

                    radius_1 = np.zeros(1)
                    for i in range(d[j].size):
                        if(normal_data[i] < base_coverage):
                            coverage = np.hstack((coverage, u[j][i]))
                            coverage1 = np.hstack((coverage1, normal_data[i]))
                    coverage = np.delete(coverage, 0)

        #             coverage1 = np.delete(coverage1, 0)
                    radius_1 = np.sum(coverage) / u[j].size
                    radius_lst = np.hstack((radius_lst, radius_1))

                    # circle = plt.Circle((cntr[j][0], cntr[j][1]),
                    #             radius_1,
                    #            facecolor='None', edgecolor= 'k')

                    rectangle = plt.Rectangle((cntr[j][0]-(radius_1/2),cntr[j][1]-(radius_1/2)),
                                              radius_1,radius_1,linewidth=1,edgecolor='k',facecolor='none')

                    # ax.add_patch(circle)
                    ax.add_patch(rectangle)
                    ax.plot(cntr[j][0], cntr[j][1], 'rs')

                    #     membership_graph(ncenters, u, xpts, ypts)

                    #         ax.set_title('Beta={0};Coverage={1:.4f}'.format(power, radius_1))
                    ax.set_title('Best Beta={:10.2f}'.format(power))
                    self.ax_chebyshev.insert(insert_index, ax)
                    insert_index+=1
            elif(image_index == 4):
                ax.plot(power_lst, intersect_num_lst)
                ax.set_title('interctect num as func of beta')
                self.ax_euclidean.insert(insert_index, ax)
                insert_index+=1

        self.fig.tight_layout()
        self.canvas.draw()
    # plt.show()


root = Tk()
app = gui(root)
root.mainloop()
