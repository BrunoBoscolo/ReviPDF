import sys
import json
import os
import threading

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QGroupBox, QRadioButton, QLabel, QLineEdit,
                             QPushButton, QTextEdit, QFileDialog, QMessageBox)
from PyQt5.QtCore import pyqtSignal, QObject

from pdf_processor import process_pdf_and_export_json, compare_pedagogic_materials, process_aulas_from_pdf

class WorkerSignals(QObject):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

class PDFProcessorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ReviPDF Processor")
        self.resize(800, 600)
        self.signals = WorkerSignals()
        self.signals.finished.connect(self._process_success)
        self.signals.error.connect(self._process_error)

        self.create_widgets()

    def create_widgets(self):
        main_layout = QVBoxLayout()

        # Mode Selection
        mode_group = QGroupBox("Select Mode")
        mode_layout = QHBoxLayout()

        self.mode_pdf = QRadioButton("Process PDF")
        self.mode_pdf.setChecked(True)
        self.mode_aulas = QRadioButton("Process Aulas")
        self.mode_compare = QRadioButton("Compare Pedagogic Materials")

        mode_layout.addWidget(self.mode_pdf)
        mode_layout.addWidget(self.mode_aulas)
        mode_layout.addWidget(self.mode_compare)
        mode_group.setLayout(mode_layout)

        self.mode_pdf.toggled.connect(self.update_file_inputs)
        self.mode_aulas.toggled.connect(self.update_file_inputs)
        self.mode_compare.toggled.connect(self.update_file_inputs)

        main_layout.addWidget(mode_group)

        # File Inputs
        file_group = QGroupBox("Select Files")
        self.file_layout = QVBoxLayout()

        # File 1 Row
        file1_row = QHBoxLayout()
        self.file1_label = QLabel("PDF File:")
        self.file1_entry = QLineEdit()
        self.file1_btn = QPushButton("Browse")
        self.file1_btn.clicked.connect(lambda: self.browse_file(self.file1_entry))

        file1_row.addWidget(self.file1_label)
        file1_row.addWidget(self.file1_entry)
        file1_row.addWidget(self.file1_btn)

        self.file_layout.addLayout(file1_row)

        # File 2 Row (Initially Hidden)
        self.file2_row_widget = QWidget()
        file2_row = QHBoxLayout(self.file2_row_widget)
        file2_row.setContentsMargins(0, 0, 0, 0)
        self.file2_label = QLabel("Student PDF:")
        self.file2_entry = QLineEdit()
        self.file2_btn = QPushButton("Browse")
        self.file2_btn.clicked.connect(lambda: self.browse_file(self.file2_entry))

        file2_row.addWidget(self.file2_label)
        file2_row.addWidget(self.file2_entry)
        file2_row.addWidget(self.file2_btn)

        self.file_layout.addWidget(self.file2_row_widget)
        self.file2_row_widget.setVisible(False)

        file_group.setLayout(self.file_layout)
        main_layout.addWidget(file_group)

        # Action Button
        self.run_btn = QPushButton("Run Processor")
        self.run_btn.setStyleSheet("background-color: green; color: white; font-weight: bold; padding: 10px;")
        self.run_btn.clicked.connect(self.run_processing)
        main_layout.addWidget(self.run_btn)

        # Status Label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: blue;")
        main_layout.addWidget(self.status_label)

        # JSON Viewer
        viewer_group = QGroupBox("Results (JSON)")
        viewer_layout = QVBoxLayout()
        self.text_area = QTextEdit()
        self.text_area.setLineWrapMode(QTextEdit.NoWrap)
        self.text_area.setReadOnly(True)
        viewer_layout.addWidget(self.text_area)
        viewer_group.setLayout(viewer_layout)
        main_layout.addWidget(viewer_group)

        self.setLayout(main_layout)

    def get_selected_mode(self):
        if self.mode_pdf.isChecked():
            return "Process PDF"
        elif self.mode_aulas.isChecked():
            return "Process Aulas"
        elif self.mode_compare.isChecked():
            return "Compare Pedagogic Materials"

    def update_file_inputs(self):
        mode = self.get_selected_mode()
        if mode == "Compare Pedagogic Materials":
            self.file1_label.setText("Teacher PDF:")
            self.file2_row_widget.setVisible(True)
        else:
            self.file1_label.setText("PDF File:")
            self.file2_row_widget.setVisible(False)

    def browse_file(self, line_edit):
        filename, _ = QFileDialog.getOpenFileName(self, "Open PDF", "", "PDF Files (*.pdf)")
        if filename:
            line_edit.setText(filename)

    def run_processing(self):
        mode = self.get_selected_mode()
        f1 = self.file1_entry.text()
        f2 = self.file2_entry.text()

        if not f1:
            QMessageBox.critical(self, "Error", "Please select the primary PDF file.")
            return

        if mode == "Compare Pedagogic Materials" and not f2:
            QMessageBox.critical(self, "Error", "Please select the student PDF file.")
            return

        if not os.path.exists(f1):
            QMessageBox.critical(self, "Error", f"File not found: {f1}")
            return

        if mode == "Compare Pedagogic Materials" and not os.path.exists(f2):
            QMessageBox.critical(self, "Error", f"File not found: {f2}")
            return

        self.run_btn.setEnabled(False)
        self.status_label.setText(f"Processing '{mode}'... Please wait.")
        self.text_area.clear()

        # Run in thread
        thread = threading.Thread(target=self._process_thread, args=(mode, f1, f2))
        thread.daemon = True
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
            self.signals.finished.emit(formatted_json)

        except Exception as e:
            self.signals.error.emit(str(e))

    def _process_success(self, formatted_json):
        self.text_area.setPlainText(formatted_json)
        self.status_label.setText("Processing Complete!")
        self.run_btn.setEnabled(True)

    def _process_error(self, error_msg):
        QMessageBox.critical(self, "Processing Error", f"An error occurred:\n{error_msg}")
        self.status_label.setText("Error during processing.")
        self.run_btn.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = PDFProcessorGUI()
    gui.show()
    sys.exit(app.exec_())
