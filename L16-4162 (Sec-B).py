from tkinter import *
from tkinter import filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import wave
from scipy.fftpack import fft
import numpy as np
from scipy.io import wavfile as wav
import sounddevice as sd

path = ""  # for music file path

class Window(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.filename = ''
        self.data = ''
        self.init_window()

    #Creation of init_window
    def init_window(self):

        # changing the title of our master widget
        self.master.title("HCI Assignment 1")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        # creating a button instance
        uploadButton = Button(self, text="Upload", command=self.fileDialog)

        # placing the button on my window
        uploadButton.place(x=80, y=520)

        # sizing the button
        uploadButton.configure(width=15,height=2)

        # creating Record Button
        recordButton = Button(self,text="Record",command=self.recordAudio)
        recordButton.place(x=210,y=520)
        recordButton.configure(width=15,height=2)

        # creating another button
        plotButton = Button(self, text="Plot FFT" , command=self.plotFFT)

        # placing the button on my window
        plotButton.place(x=580, y=520)

        # sizing the button
        plotButton.configure(width=30, height=2)

        # adding audio slider
        audioSlider = Scale(self,from_=0,to=100,orient=HORIZONTAL,length=200,command=self.moveAudioSlide)
        audioSlider.place(x=120,y=460)

        # adding FFT slider
        FFTSlider = Scale(self,from_=0,to=100, orient=HORIZONTAL, length=200,command=self.moveFFTSlide)
        FFTSlider.place(x=600, y=460)

        # adding label
        labelfont = ('times', 20, 'bold')
        label1 = Label(self,text="Audio Signal")
        label1.config(font=labelfont)
        label1.place(x=110,y=30)

        # adding another label
        label2 = Label(self, text="Fast Fourier Transform")
        label2.config(font=labelfont)
        label2.place(x=580, y=30)

        # adding label Duration
        label3 = Label(self, text="Duration in Seconds")
        label3.place(x=85,y=580)

        # adding duration text field
        entryfont = ('times',14)
        entryText = StringVar()
        entryText.set("- : --")
        durationEntry = Entry(self,text=entryText)
        durationEntry.configure(width=7)
        durationEntry.configure(font=entryfont)
        durationEntry.place(x=210,y=580)

        # ploting audio signal
        audioSignal = Figure(figsize=(5, 5), dpi=100)
        a = audioSignal.add_subplot(111)
        a.plot([0, 0], [0, 0])
        a.set_xlabel("Time")
        a.set_ylabel("Amplitude")

        audioCanvas = FigureCanvasTkAgg(audioSignal, self)
        audioCanvas.draw()
        audioCanvas.get_tk_widget().configure(width=420,height=390)
        audioCanvas.get_tk_widget().place(x=45,y=70)

        # ploting FFT
        FFT = Figure(figsize=(5,5), dpi=100)
        b = FFT.add_subplot(111)
        b.plot([0, 0], [0, 0])
        b.set_xlabel("Frequency")
        b.set_ylabel("Imaginary")

        FFTCanvas = FigureCanvasTkAgg(FFT, self)
        FFTCanvas.draw()
        FFTCanvas.get_tk_widget().configure(width=420, height=390)
        FFTCanvas.get_tk_widget().place(x=520, y=70)

    # upload file
    def fileDialog(self):
        self.filename = filedialog.askopenfilename(initialdir = "/", title = "Select a File" , filetype = (("wave" , "*.wav"),("All Files","*.")))
        path = self.filename

        if path != '':
            # ploting audio signal
            audioSignal = Figure(figsize=(5, 5), dpi=100)
            a = audioSignal.add_subplot(111)

            # setting the duration
            f = wave.open(path, 'r')
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)

            # adding duration text field
            entryfont = ('times', 14)
            entryText = StringVar()
            entryText.set(int(duration))
            durationEntry = Entry(self, text=entryText)
            durationEntry.configure(width=7)
            durationEntry.configure(font=entryfont)
            durationEntry.place(x=210,y=580)

            # plotting the graph
            rate, data = wav.read(path)
            a.plot(data[0:2000])
            a.set_xlabel("Time")
            a.set_ylabel("Amplitude")

            audioCanvas = FigureCanvasTkAgg(audioSignal, self)
            audioCanvas.draw()
            audioCanvas.get_tk_widget().configure(width=420, height=390)
            audioCanvas.get_tk_widget().place(x=45, y=70)
            self.data = ''


    def plotFFT(self):
        # plotting FFT
        if self.filename != '':
            FFT = Figure(figsize=(5, 5), dpi=100)
            b = FFT.add_subplot(111)
            rate, data = wav.read(self.filename)
            fft_out = fft(data)
            b.plot(data[0:2000], np.abs(fft_out[0:2000]))
            b.set_xlabel("Frequency")
            b.set_ylabel("Imaginary")

            FFTCanvas = FigureCanvasTkAgg(FFT, self)
            FFTCanvas.draw()
            FFTCanvas.get_tk_widget().configure(width=420, height=390)
            FFTCanvas.get_tk_widget().place(x=520, y=70)
            self.data = ''

        if self.data != '':
            FFT = Figure(figsize=(5, 5), dpi=100)
            b = FFT.add_subplot(111)
            fft_out = fft(self.data)
            b.plot(self.data, np.abs(fft_out))
            b.set_xlabel("Frequency")
            b.set_ylabel("Imaginary")

            FFTCanvas = FigureCanvasTkAgg(FFT, self)
            FFTCanvas.draw()
            FFTCanvas.get_tk_widget().configure(width=420, height=390)
            FFTCanvas.get_tk_widget().place(x=520, y=70)



    def moveAudioSlide(self,val):
        if self.filename != '':
            initialPt = int(val) * 2000
            endingPt = int(initialPt) + 2000
            audioSignal = Figure(figsize=(5, 5), dpi=100)
            a = audioSignal.add_subplot(111)

            rate, data = wav.read(self.filename)
            a.clear
            a.plot(data[initialPt:endingPt])
            a.set_xlabel("Time")
            a.set_ylabel("Amplitude")

            audioCanvas = FigureCanvasTkAgg(audioSignal, self)
            audioCanvas.draw()
            audioCanvas.get_tk_widget().configure(width=420, height=390)
            audioCanvas.get_tk_widget().place(x=45, y=70)


    def moveFFTSlide(self,val):
        if self.filename != '':
            initialPt = int(val) * 2000
            endingPt = int(initialPt) + 2000
            FFT = Figure(figsize=(5, 5), dpi=100)
            b = FFT.add_subplot(111)
            rate, data = wav.read(self.filename)
            fft_out = fft(data)
            b.clear
            b.plot(data[initialPt:endingPt], np.abs(fft_out[initialPt:endingPt]))
            b.set_xlabel("Frequency")
            b.set_ylabel("Imaginary")

            FFTCanvas = FigureCanvasTkAgg(FFT, self)
            FFTCanvas.draw()
            FFTCanvas.get_tk_widget().configure(width=420, height=390)
            FFTCanvas.get_tk_widget().place(x=520, y=70)

    def recordAudio(self):
        print("Recording")
        fs = 16000 # frequency
        d = 5 # duration
        self.data = sd.rec(int(d*fs),fs,1)
        sd.wait()

        audioSignal = Figure(figsize=(5, 5), dpi=100)
        a = audioSignal.add_subplot(111)

        # adding duration text field
        entryfont = ('times', 14)
        entryText = StringVar()
        entryText.set("5")
        durationEntry = Entry(self, text=entryText)
        durationEntry.configure(width=7)
        durationEntry.configure(font=entryfont)
        durationEntry.place(x=210, y=580)

        # plotting the graph
        a.plot(self.data)
        a.set_xlabel("Time")
        a.set_ylabel("Amplitude")

        audioCanvas = FigureCanvasTkAgg(audioSignal, self)
        audioCanvas.draw()
        audioCanvas.get_tk_widget().configure(width=420, height=390)
        audioCanvas.get_tk_widget().place(x=45, y=70)
        self.filename = ''



root = Tk()

#size of the window
root.geometry("960x640")

app = Window(root)
root.mainloop() 