class Plot:
    def __init__(self):
        pass
    
    def show_graph(self, title, x, y, fig, x_label, y_label):
    	plt.figure(fig)
		plt.title(title)
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.plot(x, y)
        plt.show()        