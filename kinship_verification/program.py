from PIL import ImageTk
import PIL.Image
import tkinter
from tkinter import *
from resizeimage import resizeimage
from tkinter import filedialog
import network as net


class Program:

    def __init__(self):
        self.root = tkinter.Tk()
        self.root.geometry("500x420")
        self.root.resizable(False, False)
        self.root.title('Kinship Verification')
        super().__init__()
        net.train_network()
        self.first_frame()
        self.root.mainloop()

    def dir_chooser(self):
        self.directory = filedialog.askdirectory(title="Select directory")

    def load_start_dataset(self, n):
        data, results, comp_results, percentage = net.test_n_pairs(n) if self.directory == '' else net.test_n_new_pairs(n, self.directory)
        self.data = data
        self.results = results
        self.comp_percentage = percentage
        self.comp_results = comp_results
        self.index = 0
        self.correct = 0
        self.data_size = len(data)

    def first_frame(self):
        self.fintro = tkinter.Frame(self.root, width=500, height=500, bg='white')
        self.fintro.pack(fill=tkinter.X)
        self.directory = ''

        # -----First frame GUI---------
        open_dir = tkinter.Button(self.fintro, text='Choose directory', bg='black', fg='white', font=("Calibri", 18),
                                      activebackground='white', activeforeground='black',
                                      command=lambda: (self.dir_chooser()))
        open_dir.place(x=140, y=50)

        question = 'Choose number of test pairs to check.'
        question_label = tkinter.Label(self.fintro, text=question, font=("Calibri", 16), bg='white')
        question_label.place(x=60, y=170)

        # DROPDOWN MENU
        variable = StringVar(self.root)
        variable.set("10")  # default value
        self.option_menu = OptionMenu(self.fintro, variable, "5", "10", "20", "50")
        self.option_menu.place(x=230, y=200)

        start = tkinter.Button(self.fintro, text='Start', bg='black', fg='white',
                                       font=("Calibri", 18),
                                       activebackground='white', activeforeground='black',
                                       command=lambda: (self.fintro.destroy(), self.load_start_dataset(int(variable.get())), self.main_frame()))
        start.place(x=210, y=300)

    def main_frame(self):
        self.f1 = tkinter.Frame(self.root, width=500, height=420, bg='white')
        self.f1.pack(fill=tkinter.X)

        first_img, second_img = self.data[self.index]

        resize = [200, 200]
        photo = PIL.Image.open(first_img)
        photo = resizeimage.resize_cover(photo, resize, validate=False)
        photo = ImageTk.PhotoImage(photo)
        photo2 = PIL.Image.open(second_img)
        photo2 = resizeimage.resize_cover(photo2, resize, validate=False)
        photo2 = ImageTk.PhotoImage(photo2)

        f_pic = tkinter.Label(self.f1, image=photo)
        f_pic.photo = photo
        f_pic.place(x=49, y=50)

        sec_pic = tkinter.Label(self.f1, image=photo2)
        sec_pic.photo = photo2
        sec_pic.place(x=250, y=50)

        text = 'Image: ' + str(self.index + 1) + '/' + str(self.data_size)
        current_image_label = tkinter.Label(self.f1, text=text, font=("Calibri", 16), bg='white')
        current_image_label.place(x=190, y=10)

        question = 'Do you think these two persons are related?'
        question_label = tkinter.Label(self.f1, text=question, font=("Calibri", 16), bg='white')
        question_label.place(x=60, y=260)

        self.yes_btn = tkinter.Button(self.f1, text='Yes!', bg='green', fg='#ffffff', font=("Calibri", 18),
                                      activebackground='#ffffff', activeforeground='green',
                                      command=lambda: (self.next_picture(1.)))
        self.yes_btn.place(x=187, y=300)

        self.no_btn = tkinter.Button(self.f1, text='No!', bg='#ff0000', fg='#ffffff', font=("Calibri", 18),
                                     activebackground='#ffffff', activeforeground='#ff0000',
                                     command=lambda: (self.next_picture(0.)))
        self.no_btn.place(x=255, y=300)

        self.you_label = tkinter.Label(self.f1, text='You', font=("Calibri", 15), bg='white')
        self.you_label.place(x=115, y=370)

        self.comp_label = tkinter.Label(self.f1, text='Computer', font=("Calibri", 15), bg='white')
        self.comp_label.place(x=195, y=370)

        self.corr_label = tkinter.Label(self.f1, text='Correct', font=("Calibri", 15), bg='white')
        self.corr_label.place(x=315, y=370)

    def next_picture(self, answer):
        # update corr_label
        if self.results[self.index] == 1:
            self.corr_label.config(text='Correct: YES', bg='green')
        else:
            self.corr_label.config(text='Correct: NO', bg='red')

        # update comp_label
        if self.comp_results[self.index] == 1:
            self.comp_label.configure(bg='green')
        else:
            self.comp_label.configure(bg='red')

        # update human answers
        if answer == 1:
            self.you_label.configure(bg='green')
        else:
            self.you_label.configure(bg='red')

        if answer == self.results[self.index]:
            self.correct += 1

        self.yes_btn.destroy()
        self.no_btn.destroy()

        self.next_btn = tkinter.Button(self.f1, text='Next!', bg='black', fg='white', font=("Calibri", 18),
                                       activebackground='white', activeforeground='black',
                                       command=lambda: (self.next_pair_action()))
        self.next_btn.place(x=200, y=300)

        self.root.update()

    def next_pair_action(self):
        self.f1.destroy()

        self.index += 1
        if self.index == self.data_size:
            self.end_frame()
        else:
            self.main_frame()

    def end_frame(self):
        self.f1 = tkinter.Frame(self.root, width=600, height=420, bg='white')
        self.f1.pack(fill=tkinter.X)

        result = 'Computer success percentage: ' + "{0:.2f}%".format(100.0 * self.comp_percentage)
        last = tkinter.Label(self.f1, text=result, font=("Calibri", 15), fg='black', bg='white')
        last.place(x=80, y=100)

        result = 'Your success percentage: ' + "{0:.2f}%".format(100.0 * self.correct / self.data_size)
        last = tkinter.Label(self.f1, text=result, font=("Calibri", 15), fg='black', bg='white')
        last.place(x=110, y=200)

        if self.comp_percentage < self.correct / self.data_size:
            better_worse = 'You are better than computer :)'
            background = 'green'

        elif self.comp_percentage == self.correct / self.data_size:
            better_worse = 'You guessed the same as the computer :|'
            background = 'yellow'

        else:
            better_worse = 'You are worse than computer :\'('
            background = 'red'

        last = tkinter.Label(self.f1, text=better_worse, font=("Calibri", 15), bg='white', fg=background)
        last.place(x=110, y=300)


if __name__ == '__main__':
    Program()

