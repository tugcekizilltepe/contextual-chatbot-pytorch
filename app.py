from tkinter import *
import tkinter.font
from chatbot_lib.chat import get_response
from settings.tkinter_config import BOT_NAME, background_config, text_box_config, title_config, button_config


class ChatApplication:
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("Konuşma Penceresi")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=470, height=550, bg=background_config["COLOR"])

        TEXT_BOX_FONT = tkinter.font.Font(family=text_box_config['TEXT_FONT_FAMILY'], size=15, weight="normal")
        BUTTON_TEXT_FONT = tkinter.font.Font(family=button_config['TEXT_FONT_FAMILY'], size=15, weight="bold")
        TITLE_TEXT_FONT = tkinter.font.Font(family=title_config['TEXT_FONT_FAMILY'], size=14, weight="bold")

        # Header
        head_label = Label(self.window, bg=background_config["COLOR"], fg=background_config["TEXT_COLOR"],
                           text="Hoşgeldin!", font=TITLE_TEXT_FONT, pady=10)

        head_label.place(relwidth=1)

        # Set text widget
        self.text_widget = Text(self.window, width=20, height=2, bg=text_box_config["COLOR"],
                                fg=text_box_config["TEXT_COLOR"], font=TEXT_BOX_FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        # Set scrollbar
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview())

        # Text box label
        text_box_label = Label(self.window, bg=background_config["TEXT_BOX"], height=80)
        text_box_label.place(relwidth=1, rely=0.825)

        # Message entry box
        self.message_entry = Entry(text_box_label, bg=text_box_config["COLOR"], fg=text_box_config["TEXT_COLOR"],
                                   font=TEXT_BOX_FONT)
        self.message_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.message_entry.focus()
        self.message_entry.bind("<Return>", self._on_enter_pressed)

        # Set button
        button = Button(text_box_label, text="Yolla", font=BUTTON_TEXT_FONT, width=20, bg=button_config["COLOR"]
                        , command=lambda: self._on_enter_pressed(None))

        button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

    def _on_enter_pressed(self, event):
        msg = self.message_entry.get()
        self._insert_message(msg, "Sen")

    def _insert_message(self, msg, sender):
        if not msg:
            return

        self.message_entry.delete(0, END)
        incoming_text = f"\n\n{sender}:\n{msg}"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, incoming_text)
        self.text_widget.configure(state=DISABLED)

        response = f"\n\n{BOT_NAME}:\n{get_response(incoming_text)}"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, response)
        self.text_widget.configure(state=DISABLED)

        self.text_widget.see(END)


if __name__ == "__main__":
    app = ChatApplication()
    app.run()
