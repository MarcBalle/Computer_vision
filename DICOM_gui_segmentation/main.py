from tkinter import *
from tkinter.filedialog import askdirectory
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from skimage import measure
import pydicom
import cv2
import os
import glob
import sys
import copy


class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        self.title("PIM GUI")
        self.minsize(640, 400)
        self.wm_iconbitmap('icon.ico')

        self.dcm_series = []  # Array de datos
        self.raw_img = []  # Objeto Pydicom
        self.contours = [] # Lista que almacenará los contornos del tumor en cada corte
        self.load_dcm()  # cargamos DICOM series
        self.slice = int(len(self.dcm_series) / 2)  # slice de en medio
        self.matplotcanvas()

        self.point_tumor = (355, 205, 42)  # punto de referencia del tumor

        # Inicialización de los widgets
        self.slice_var = IntVar()
        self.slice_var.set(int(len(self.dcm_series) / 2))  # slice del medio
        self.slicer = Spinbox(self, from_=0, to=len(self.dcm_series), width=5, textvariable=self.slice_var)
        self.slicer.pack(side=LEFT)

        self.slicer_button = Button(self, text="View Slice", command=self.print_slice)
        self.slicer_button.pack(side=LEFT)

        self.y_var = IntVar()
        self.y_var.set(0)  # pixel origen
        self.y_spin = Spinbox(self, from_=0, to=len(self.dcm_series), width=5, textvariable=self.y_var)
        self.y_spin.pack(side=LEFT)

        self.x_var = IntVar()
        self.x_var.set(0)  # pixel origen
        self.x_spin = Spinbox(self, from_=0, to=len(self.dcm_series), width=5, textvariable=self.x_var)
        self.x_spin.pack(side=LEFT)

        self.pixel_button = Button(self, text="Get Pixel Value", command=self.get_pixel_value)
        self.pixel_button.pack(side=LEFT)

        self.pixel_label = Label(self, text="None")
        self.pixel_label.pack(side=LEFT)

        self.header_button = Button(self, text="Show Header", command=self.show_header)
        self.header_button.pack(side=LEFT, padx=3)

        self.log_button = Button(self, text="Log", command=self.log_image)
        self.log_button.pack(side=LEFT, padx=3)

        self.eq_button = Button(self, text="Histogram Equalization", command=self.histogram_equalization)
        self.eq_button.pack(side=LEFT)

        self.clear_contrast_button = Button(self, text="Clear Contrast", command=self.clear_contrast)
        self.clear_contrast_button.pack(side=LEFT)

        self.level_var = DoubleVar()
        self.level_var.set(0.6)
        self.slicer = Spinbox(self, from_=0, to=1, width=5, format='%.2f', increment=0.05, textvariable=self.level_var)
        self.slicer.pack(side=LEFT, padx=3)

        self.seg_button = Button(self, text="Segment Tumor", command=self.segment_tumor)
        self.seg_button.pack(side=LEFT)

        self.clear_button = Button(self, text="Clear Image", command=self.clear_contour)
        self.clear_button.pack(side=RIGHT)


    def load_dcm(self):
        ''' Se cargan las imágenes en una lista'''
        file = askdirectory()
        for filename in sorted(glob.glob(os.path.join(file, '*.dcm')), key=key_func):
            img = pydicom.dcmread(filename)
            self.dcm_series.append(np.flip(img.pixel_array, axis=0)) #matriz de píxeles
            self.raw_img.append(img) # objeto Pydicom

    def matplotcanvas(self):
        ''' Inicialización del entorno en el que se muestran los cortes'''
        self.f = Figure(figsize=(5, 5), dpi=100)
        self.a = self.f.add_subplot(111)
        self.a.imshow(self.dcm_series[self.slice], cmap=plt.cm.get_cmap('bone'))

        self.canvas = FigureCanvasTkAgg(self.f, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, self) # comandos que permiten el zoom, entre otros
        self.canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=True)

    def print_slice(self):
        ''' Función que permite plotear un corte cada vez que se cambia de corte, se modifica el contraste,
        se calcula una segmentación o cualquier otro tipo de cambio. '''
        self.a.cla()  # se limpia el plot cada vez que una nueva modificación se realiza
        self.slice = self.slice_var.get()
        self.a.imshow(self.dcm_series[self.slice], cmap=plt.cm.get_cmap('bone'))
        self.canvas.draw()
        index_contour = self.slice - self.point_tumor[2]  # primera slice del tumor
        if len(self.contours) > 0 and index_contour >= 0: # si hay contornos calculados y el corte actual es igual o mayor a 42
            try:  # si index_contour se encuentra en los corte con tumor
                self.a.plot(self.contours[index_contour][:, 1], self.contours[index_contour][:, 0], 'r', linewidth=1)
                self.canvas.draw()
            except: # si no no se hace nada
                pass

    def get_pixel_value(self):
        ''' FUnción que lee las coordenadas de un píxel introducidas y muestra el valor de este'''
        col = self.y_var.get()
        row = self.x_var.get()

        pixel_value = self.dcm_series[self.slice][row, col]
        text = StringVar()
        text.set(str(pixel_value))
        self.pixel_label.config(textvariable=text)

    def show_header(self):
        ''' Función para mostrar el encabezado del corte actual en una nueva ventana'''
        sys.stdout = open('header.txt', 'w') # se redirige la salida estándar al fichero txt
        print(self.raw_img[self.slice]) # es escribe el encabezado en el fichero
        sys.stdout.close()
        f = open("header.txt") # se abre el fichero escrito
        top = Toplevel()  # nueva ventana
        top.title("DICOM Header")
        t = Text(top, height=400, width=400)
        t.pack(fill=BOTH, expand=True)
        t.insert(1.0, f.read()) # se escribe el encabezado en la nueva ventana

    def segment_tumor(self):
        ''' Función que permite segmentar el tumor con isocontornos'''
        mask = np.zeros(self.dcm_series[0].shape)
        # Máscara binaria donde un cuadrado 60x60 englobará al tumor
        mask[self.point_tumor[1] - 30:self.point_tumor[1] + 30, self.point_tumor[0] - 30:self.point_tumor[0] + 30] = 1.0
        # Se encuentran los contornos para cada corte
        for slice_ix in range(self.point_tumor[2], self.point_tumor[2] + 7):  # desde la slice 42 a la 48
            img_masked = self.dcm_series[slice_ix] * mask # únicamente se muestra la información del tumor y su entorno
            level = (img_masked.max() - img_masked.min()) * self.level_var.get() # nivel de gris para los isocontornos
            contour = measure.find_contours(img_masked, level=level)
            shapes = [cont.shape[0] for cont in contour] #número de puntos por cada contorno encontrado
            index = np.argsort(np.array(shapes))[::-1]  # el contorno con más puntos es el correspondiente a la lesión
            self.contours.append(contour[index[0]])

        self.print_slice()  # se plotean los contornos

    def clear_contour(self):
        ''' Función que permite limpiar los contornos de la imagen'''
        self.contours = []
        self.print_slice()

    def log_image(self):
        ''' Función que calcula el logarimto natural de la image. Realce de los detalles de bajo contraste'''
        self.dcm_series_cp = copy.deepcopy(self.dcm_series)  # se copia la imagen original para después reestablecerla.
        for i in range(len(self.dcm_series)):
            img_aux = self.dcm_series[i] - self.dcm_series[i].min() + 1 # Valor mínimo del corte igual a 1
            self.dcm_series[i] = np.log(img_aux)

        self.print_slice()  # se plotea la imagen transformada

    def histogram_equalization(self):
        ''' Ecualización del histograma'''
        self.dcm_series_cp = copy.deepcopy(self.dcm_series) # se copia la imagen original para después reestablecerla.
        for i in range(len(self.dcm_series)):
            img_flattened = self.dcm_series[i].flatten()
            # Se cambia el rango dinámico de la imagen para trabajar con 8 bits
            img_255 = ((self.dcm_series[i] - img_flattened.min()) / (img_flattened.max() - img_flattened.min())) * 255
            img_eq = cv2.equalizeHist(img_255.astype('uint8'))
            self.dcm_series[i] = img_eq

        self.print_slice()  # se plotea la imagen transformada


    def clear_contrast(self):
        ''' Función que permite reestablecer la imagen original antes del ajuste de contraste'''
        self.dcm_series = copy.deepcopy(self.dcm_series_cp) # se copia la copia ya hecha antes del ajuste de contraste
        self.print_slice()  # se plotea la imagen original



def key_func(x):
    return os.path.split(x)[-1]


def main():
    root = Root()
    root.mainloop()


if __name__ == "__main__":
    main()
