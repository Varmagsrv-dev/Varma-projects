import sqlite3
import tkinter as tk
from tkinter import messagebox

# ==========================
# DATABASE
# ==========================

conn = sqlite3.connect("students.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS students(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER,
    course TEXT,
    marks REAL
)
""")

conn.commit()


# ==========================
# FUNCTIONS
# ==========================

def add_student():

    name = entry_name.get()
    age = entry_age.get()
    course = entry_course.get()
    marks = entry_marks.get()

    if name == "":
        messagebox.showerror("Error", "Name cannot be empty")
        return

    cursor.execute("""
    INSERT INTO students(name, age, course, marks)
    VALUES (?, ?, ?, ?)
    """, (name, age, course, marks))

    conn.commit()

    messagebox.showinfo("Success", "Student Added")

    entry_name.delete(0, tk.END)
    entry_age.delete(0, tk.END)
    entry_course.delete(0, tk.END)
    entry_marks.delete(0, tk.END)

    show_students()


def show_students():

    listbox.delete(0, tk.END)

    cursor.execute("SELECT * FROM students")

    students = cursor.fetchall()

    for student in students:
        listbox.insert(tk.END, student)


# ==========================
# GUI
# ==========================

root = tk.Tk()

root.title("Student Management System")
root.geometry("700x500")

# Labels

tk.Label(root, text="Name").pack()

entry_name = tk.Entry(root, width=40)
entry_name.pack()

tk.Label(root, text="Age").pack()

entry_age = tk.Entry(root, width=40)
entry_age.pack()

tk.Label(root, text="Course").pack()

entry_course = tk.Entry(root, width=40)
entry_course.pack()

tk.Label(root, text="Marks").pack()

entry_marks = tk.Entry(root, width=40)
entry_marks.pack()

# Buttons

btn_add = tk.Button(
    root,
    text="Add Student",
    command=add_student
)

btn_add.pack(pady=10)

btn_view = tk.Button(
    root,
    text="Refresh Students",
    command=show_students
)

btn_view.pack()

# Listbox

listbox = tk.Listbox(
    root,
    width=100,
    height=15
)

listbox.pack(pady=20)

show_students()

root.mainloop()

conn.close()
