import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import json
import os

from pdf_processor import process_pdf_and_export_json, compare_pedagogic_materials, process_aulas_from_pdf

class PDFProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ReviPDF Processor")
        self.root.geometry("800x600")

        self.mode_var = tk.StringVar(value="Process PDF")

        self.file1_path = tk.StringVar()
        self.file2_path = tk.StringVar()

        self.create_widgets()

    def create_widgets(self):
        # Mode Selection
        mode_frame = tk.LabelFrame(self.root, text="Select Mode", padx=10, pady=10)
        mode_frame.pack(fill="x", padx=10, pady=5)

        modes = ["Process PDF", "Process Aulas", "Compare Pedagogic Materials"]
        for mode in modes:
            tk.Radiobutton(mode_frame, text=mode, variable=self.mode_var, value=mode, command=self.update_file_inputs).pack(side="left", padx=5)

        # File Inputs
        self.file_frame = tk.LabelFrame(self.root, text="Select Files", padx=10, pady=10)
        self.file_frame.pack(fill="x", padx=10, pady=5)

        self.file1_label = tk.Label(self.file_frame, text="PDF File:")
        self.file1_label.grid(row=0, column=0, sticky="w")
        self.file1_entry = tk.Entry(self.file_frame, textvariable=self.file1_path, width=50)
        self.file1_entry.grid(row=0, column=1, padx=5)
        self.file1_btn = tk.Button(self.file_frame, text="Browse", command=lambda: self.browse_file(self.file1_path))
        self.file1_btn.grid(row=0, column=2)

        self.file2_label = tk.Label(self.file_frame, text="Student PDF:")
        self.file2_entry = tk.Entry(self.file_frame, textvariable=self.file2_path, width=50)
        self.file2_btn = tk.Button(self.file_frame, text="Browse", command=lambda: self.browse_file(self.file2_path))

        self.update_file_inputs()

        # Action Button
        self.run_btn = tk.Button(self.root, text="Run Processor", command=self.run_processing, bg="green", fg="white", font=("Arial", 12, "bold"))
        self.run_btn.pack(pady=10)

        # Status Label
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_label = tk.Label(self.root, textvariable=self.status_var, fg="blue")
        self.status_label.pack()

        # JSON Viewer
        viewer_frame = tk.LabelFrame(self.root, text="Results (JSON)", padx=10, pady=10)
        viewer_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.text_area = tk.Text(viewer_frame, wrap="none")
        self.text_area.pack(side="left", fill="both", expand=True)

        scrollbar_y = tk.Scrollbar(viewer_frame, orient="vertical", command=self.text_area.yview)
        scrollbar_y.pack(side="right", fill="y")
        self.text_area.configure(yscrollcommand=scrollbar_y.set)

        scrollbar_x = tk.Scrollbar(viewer_frame, orient="horizontal", command=self.text_area.xview)
        scrollbar_x.pack(side="bottom", fill="x")
        self.text_area.configure(xscrollcommand=scrollbar_x.set)

    def update_file_inputs(self):
        mode = self.mode_var.get()
        if mode == "Compare Pedagogic Materials":
            self.file1_label.config(text="Teacher PDF:")
            self.file2_label.grid(row=1, column=0, sticky="w", pady=5)
            self.file2_entry.grid(row=1, column=1, padx=5, pady=5)
            self.file2_btn.grid(row=1, column=2, pady=5)
        else:
            self.file1_label.config(text="PDF File:")
            self.file2_label.grid_forget()
            self.file2_entry.grid_forget()
            self.file2_btn.grid_forget()

    def browse_file(self, path_var):
        filename = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if filename:
            path_var.set(filename)

    def run_processing(self):
        mode = self.mode_var.get()
        f1 = self.file1_path.get()
        f2 = self.file2_path.get()

        if not f1:
            messagebox.showerror("Error", "Please select the primary PDF file.")
            return

        if mode == "Compare Pedagogic Materials" and not f2:
            messagebox.showerror("Error", "Please select the student PDF file.")
            return

        if not os.path.exists(f1):
            messagebox.showerror("Error", f"File not found: {f1}")
            return

        if mode == "Compare Pedagogic Materials" and not os.path.exists(f2):
            messagebox.showerror("Error", f"File not found: {f2}")
            return

        self.run_btn.config(state="disabled")
        self.status_var.set(f"Processing '{mode}'... Please wait.")
        self.text_area.delete(1.0, tk.END)

        # Run in thread
        thread = threading.Thread(target=self._process_thread, args=(mode, f1, f2))
        thread.start()

    def _process_thread(self, mode, f1, f2):
        try:
            result_data = None
            if mode == "Process PDF":
                result_data = process_pdf_and_export_json(f1, "response.json")
            elif mode == "Process Aulas":
                result_data = process_aulas_from_pdf(f1, "aulas_report.json")
            elif mode == "Compare Pedagogic Materials":
                result_data = compare_pedagogic_materials(f1, f2, "comparison_response.json")

            formatted_json = json.dumps(result_data, indent=4, ensure_ascii=False)

            # Update GUI from main thread
            self.root.after(0, self._process_success, formatted_json)
        except Exception as e:
            self.root.after(0, self._process_error, str(e))

    def _process_success(self, formatted_json):
        self.text_area.insert(tk.END, formatted_json)
        self.status_var.set("Processing Complete!")
        self.run_btn.config(state="normal")

    def _process_error(self, error_msg):
        messagebox.showerror("Processing Error", f"An error occurred:\n{error_msg}")
        self.status_var.set("Error during processing.")
        self.run_btn.config(state="normal")

if __name__ == "__main__":
    root = tk.Tk()
    app = PDFProcessorGUI(root)
    root.mainloop()
