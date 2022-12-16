# Python 3.x code
# Imports
import tkinter, time

class Popup:
    
    def __init__(self, pred_class=0, timeout = 1):
        self.timeout = timeout
        # This code is to hide the main tkinter window
        self.root = tkinter.Tk()
        # root.attributes("-fullscreen", True)
        self.root.geometry('1600x700')
        #root.option_add('*font', 'Roboto -65')
        # Message Box
        if pred_class == 0:
            self.root.title("Results")
            label = tkinter.Label(self.root, text="0 : No Grasp")
            label.config(font=("Courier", 120))
            label.pack(side="top", fill="both", expand=True, padx=5, pady=5)

        elif pred_class == 1:
            self.root.title("Results")
            label = tkinter.Label(self.root, text="1 : Unstable Grasp")
            label.config(font=("Courier", 120))
            label.pack(side="top", fill="both", expand=True, padx=5, pady=5)

        else:
            self.root.title("Results")
            label = tkinter.Label(self.root, text="2 : Stable Grasp")
            label.config(font=("Courier", 120))
            label.pack(side="top", fill="both", expand=True, padx=5, pady=5)   

        button = tkinter.Button(self.root, text="OK", command=lambda: self.root.destroy())
        button.pack(side="bottom", fill="both", expand=False)
        self.root.after(self.timeout*5000,lambda:self.root.destroy())
        self.root.mainloop()