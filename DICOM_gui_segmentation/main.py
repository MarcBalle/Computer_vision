from tkinter import *
from tkinter.filedialog import askdirectory
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from skimage import measure
import pydicom
import os
import glob
import sys


class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        self.title("PIM GUI")
        self.minsize(640, 400)
        self.wm_iconbitmap('icon.ico')

        self.dcm_series = []  # Array de datos
        self.raw_img = []  # Objeto pydicom
        self.contours = []
        self.load_dcm()  # cargamos DICOM series
        self.slice = int(len(self.dcm_series) / 2)  # slice de en medio
        self.matplotcanvas()

        self.point_tumor = (355, 205, 42)  # lesion center

        self.slice_var = IntVar()
        self.slice_var.set(int(len(self.dcm_series) / 2))  # slice del medio
        self.slicer = Spinbox(self, from_=0, to=len(self.dcm_series), width=5, textvariable=self.slice_var)
        self.slicer.pack(side=LEFT)

        self.slicer_button = Button(self, text="View Slice", command=self.print_slice)
        self.slicer_button.pack(side=LEFT)

        self.x_var = IntVar()
        self.x_var.set(0)  # pixel origen
        self.x_spin = Spinbox(self, from_=0, to=len(self.dcm_series), width=5, textvariable=self.x_var)
        self.x_spin.pack(side=LEFT)

        self.y_var = IntVar()
        self.y_var.set(0)  # pixel origen
        self.y_spin = Spinbox(self, from_=0, to=len(self.dcm_series), width=5, textvariable=self.y_var)
        self.y_spin.pack(side=LEFT)

        self.pixel_button = Button(self, text="Get Pixel Value", command=self.get_pixel_value)
        self.pixel_button.pack(side=LEFT)

        self.pixel_label = Label(self, text="None")
        self.pixel_label.pack(side=LEFT)

        self.header_button = Button(self, text="Show Header", command=self.show_header)
        self.header_button.pack(side=TOP)

        self.clear_button = Button(self, text="Clear Image", command=self.clear_contour)
        self.clear_button.pack(side=RIGHT)

        self.seg_button = Button(self, text="Segment Tumor", command=self.segment_tumor)
        self.seg_button.pack(side=RIGHT)


    def load_dcm(self):
        file = askdirectory()
        for filename in sorted(glob.glob(os.path.join(file, '*.dcm')), key=key_func):
            img = pydicom.dcmread(filename)
            self.dcm_series.append(np.flip(img.pixel_array, axis=0))
            self.raw_img.append(img)

    def matplotcanvas(self):
        self.f = Figure(figsize=(5, 5), dpi=100)
        self.a = self.f.add_subplot(111)
        self.a.imshow(self.dcm_series[self.slice], cmap=plt.cm.get_cmap('bone'))

        self.canvas = FigureCanvasTkAgg(self.f, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=True)

    def print_slice(self):
        self.a.cla()  # clean axes every time a new slice is plotted
        self.slice = self.slice_var.get()
        self.a.imshow(self.dcm_series[self.slice], cmap=plt.cm.get_cmap('bone'))
        self.canvas.draw()
        index_contour = self.slice - self.point_tumor[2]  # primera slice del tumor
        if len(self.contours) > 0 and index_contour >= 0:
            try:  # si la slice tiene tumor, se plotea su contorno; por el contrario se deja igual
                self.a.plot(self.contours[index_contour][:, 1], self.contours[index_contour][:, 0], 'r', linewidth=1)
                self.canvas.draw()
            except:
                pass
        #self.canvas.draw()

    def get_pixel_value(self):
        x_coord = self.x_var.get()
        y_coord = self.y_var.get()

        pixel_value = self.dcm_series[self.slice][x_coord, y_coord]
        text = StringVar()
        text.set(str(pixel_value))
        self.pixel_label.config(textvariable=text)

    def show_header(self):
        sys.stdout = open('header.txt', 'w')
        print(self.raw_img[self.slice])
        sys.stdout.close()
        f = open("header.txt")
       # self.T.insert(1.0, f.read())
        top = Toplevel()  # otra ventana para leer el encabezado
        top.title("DICOM Header")
        t = Text(top, height=400, width=400)
        t.pack(fill=BOTH, expand=True)
        t.insert(1.0, f.read())

    def segment_tumor(self):
        mask = np.zeros(self.dcm_series[0].shape)
        mask[self.point_tumor[1] - 30:self.point_tumor[1] + 30, self.point_tumor[0] - 30:self.point_tumor[0] + 30] = 1.0
        # Find contours for each slice
        for slice_ix in range(self.point_tumor[2], self.point_tumor[2] + 7):  # from slice 42 to 48
            img_masked = self.dcm_series[slice_ix] * mask
            level = (img_masked.max() - img_masked.min()) * 0.6
            contour = measure.find_contours(img_masked, level=level)
            shapes = [cont.shape[0] for cont in contour]
            index = np.argsort(np.array(shapes))[::-1]  # el contorno con más puntos es el correspondiente a la lesión
            self.contours.append(contour[index[0]])

        self.print_slice()

    def clear_contour(self):
        self.contours = []
        self.print_slice()


def key_func(x):
    return os.path.split(x)[-1]


def main():
    root = Root()
    root.mainloop()


if __name__ == "__main__":
    main()
