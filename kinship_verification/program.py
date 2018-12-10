import PIL.ImageTk, PIL.Image
import cv2
import time
from tkinter import *


class Program:

    def __init__(self, data, results,comp_results, percentage):
        self.root = Tk()
        self.root.geometry("%dx%d%+d%+d" % (600, 400, 300, 200))
        self.root.title('Kinship Verification')
        super().__init__()
        self.data = data
        self.results = results
        self.comp_percentage = percentage
        self.comp_results = comp_results
        self.index = 0
        self.correct = 0
        self.data_size = len(data)
        self.main_frame()
        self.root.mainloop()

    def main_frame(self):
        self.f1 = Frame(self.root, width=600, height=400)
        self.f1.pack(fill=X)

        first_img, second_img = self.data[self.index]

        first_img = cv2.cvtColor(cv2.imread(first_img), cv2.COLOR_BGR2RGB)
        second_img = cv2.cvtColor(cv2.imread(second_img), cv2.COLOR_BGR2RGB)

        photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(first_img))
        photo2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(second_img))

        f_pic = Label(self.f1, image=photo)
        f_pic.photo = photo
        f_pic.pack(side=LEFT)

        sec_pic = Label(self.f1, image=photo2)
        sec_pic.photo = photo2
        sec_pic.pack(side=RIGHT)

        text = 'Image: ' + str(self.index+1) + '/' + str(self.data_size)
        aa = Label(self.f1, text=text, font=("Times", 15))
        aa.pack(side=TOP)

        yes_btn = Button(self.f1, background='RoyalBlue1', font=("Times", 15), text='Yes',
                   command=lambda: (self.next_picture(1.)))
        yes_btn.pack(side=TOP)

        no_btn = Button(self.f1, background='RoyalBlue1', font=("Times", 15), text='No',
                   command=lambda: (self.next_picture(0.)))
        no_btn.pack(side=TOP)

        self.you_label = Label(self.f1, text='You', font=("Times", 15), padx= 50)
        self.you_label.pack(side=LEFT)

        self.comp_label = Label(self.f1, text='Computer', font=("Times", 15), padx= 50)
        self.comp_label.pack(side=LEFT)

        self.corr_label = Label(self.f1, text='Correct', font=("Times", 15), padx= 50)
        self.corr_label.pack(side=LEFT)



    def next_picture(self, answer):
        # update corr_label
        if self.results[self.index] == 1:
            self.corr_label.config(text='Correct: YES')
        else:
            self.corr_label.config(text='Correct: NO')

        # update comp_label
        if self.results[self.index] == self.comp_results[self.index]:
            self.comp_label.configure(bg='green')
        else:
            self.comp_label.configure(bg='red')

        # update human answers
        if answer == self.results[self.index]:
            self.correct += 1
            self.you_label.configure(bg='green')
        else:
            self.you_label.configure(bg='red')
        self.root.update()
        time.sleep(2.5)
        self.f1.destroy()

        self.index += 1
        if self.index == self.data_size:
            self.end_frame()
        else:
            self.main_frame()

    def end_frame(self):
        self.f1 = Frame(self.root, width=600, height=400)
        self.f1.pack(fill=X)

        result = 'Computer: ' + "{0:.2f}".format(self.comp_percentage)
        last = Label(self.f1, text=result, font=("Times", 15), pady=20)
        last.pack(side=BOTTOM)

        result = 'You: ' + "{0:.2f}".format(self.correct / self.data_size)
        last = Label(self.f1, text=result, font=("Times", 15), pady=20)
        last.pack(side=BOTTOM)
