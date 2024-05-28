# Sample code dùng PyQt5 UI tạo giao diện ứng dụng xử lý lưu trữ dữ liệu với SQLite.

import sys
import json
import sqlite3
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QLabel, QLineEdit, QListWidget, QListWidgetItem, QComboBox, QMessageBox, QSizePolicy
from PyQt5.QtCore import Qt, QMimeData
from PyQt5.QtGui import QDrag

class DatabaseManager:
    def __init__(self, db_name='actions.db'):
        self.conn = sqlite3.connect(db_name)
        self.create_table()
        self.check_and_add_columns()

    def create_table(self):
        query = '''
        CREATE TABLE IF NOT EXISTS templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            actions TEXT NOT NULL
        )
        '''
        self.conn.execute(query)
        self.conn.commit()

    def check_and_add_columns(self):
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA table_info(templates)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'name' not in columns:
            cursor.execute("ALTER TABLE templates ADD COLUMN name TEXT")
        self.conn.commit()

    def save_template(self, name, actions):
        actions_json = json.dumps(actions)
        query = 'INSERT INTO templates (name, actions) VALUES (?, ?)'
        self.conn.execute(query, (name, actions_json))
        self.conn.commit()

    def update_template(self, template_id, name, actions):
        actions_json = json.dumps(actions)
        query = 'UPDATE templates SET name = ?, actions = ? WHERE id = ?'
        self.conn.execute(query, (name, actions_json, template_id))
        self.conn.commit()

    def load_templates(self):
        query = 'SELECT id, name, actions FROM templates'
        cursor = self.conn.execute(query)
        templates = [{'id': row[0], 'name': row[1], 'actions': self.ensure_action_names(json.loads(row[2]))} for row in cursor]
        return templates

    def ensure_action_names(self, actions):
        for action in actions:
            if 'Name' not in action:
                action['Name'] = 'Unnamed Action'
        return actions

    def delete_template(self, template_id):
        query = 'DELETE FROM templates WHERE id = ?'
        self.conn.execute(query, (template_id,))
        self.conn.commit()

class DraggableListWidget(QListWidget):
    def __init__(self):
        super().__init__()
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setSelectionMode(QListWidget.SingleSelection)

    def startDrag(self, supportedActions):
        item = self.currentItem()
        if item:
            drag = QDrag(self)
            mime_data = QMimeData()
            mime_data.setText(item.text())
            drag.setMimeData(mime_data)
            drag.exec_(Qt.MoveAction)

    def dropEvent(self, event):
        source_row = self.currentRow()
        target_row = self.indexAt(event.pos()).row()
        if target_row == -1:
            target_row = self.count() - 1
        if source_row != target_row:
            item = self.takeItem(source_row)
            self.insertItem(target_row, item)
        self.reorder_steps()
        event.accept()

    def reorder_steps(self):
        for i in range(self.count()):
            item = self.item(i)
            item_text = item.text()
            item.setText(f"{i + 1}: {item_text.split(': ')[1]}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Action Template Manager')
        self.setGeometry(100, 100, 1000, 600)

        self.db_manager = DatabaseManager()

        main_layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # Template List
        self.template_list = QListWidget()
        self.template_list.itemClicked.connect(self.load_template)
        left_layout.addWidget(QLabel('Templates:'))
        left_layout.addWidget(self.template_list)

        # Action Form
        form_layout = QVBoxLayout()

        self.action_name_input = QLineEdit(self)
        form_layout.addWidget(QLabel('Action Name:'))
        form_layout.addWidget(self.action_name_input)

        self.type_input = QComboBox(self)
        self.type_input.addItems(['image', 'path', 'xpath', 'element id', 'class', 'app'])
        form_layout.addWidget(QLabel('Type:'))
        form_layout.addWidget(self.type_input)

        self.type_target_input = QLineEdit(self)
        form_layout.addWidget(QLabel('Target by type: application name if type is app. Otherwise, it is a processing link.'))
        form_layout.addWidget(self.type_target_input)

        self.media_input = QLineEdit(self)
        form_layout.addWidget(QLabel('Media (if image):'))
        form_layout.addWidget(self.media_input)

        self.content_input = QLineEdit(self)
        form_layout.addWidget(QLabel('Content:'))
        form_layout.addWidget(self.content_input)

        button_layout = QHBoxLayout()
        self.add_button = QPushButton('Add Action', self)
        self.add_button.clicked.connect(self.add_action)
        self.add_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        button_layout.addWidget(self.add_button)

        self.update_button = QPushButton('Update Action', self)
        self.update_button.clicked.connect(self.update_action)
        self.update_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        button_layout.addWidget(self.update_button)

        form_layout.addLayout(button_layout)
        right_layout.addLayout(form_layout)

        # Action List
        self.action_list = DraggableListWidget()
        self.action_list.itemDoubleClicked.connect(self.edit_action)
        right_layout.addWidget(QLabel('Actions:'))
        right_layout.addWidget(self.action_list)

        self.template_name_input = QLineEdit(self)
        right_layout.addWidget(QLabel('Template File Name:'))
        right_layout.addWidget(self.template_name_input)

        template_button_layout = QHBoxLayout()
        self.save_button = QPushButton('Save Template', self)
        self.save_button.clicked.connect(self.save_template)
        self.save_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        template_button_layout.addWidget(self.save_button)

        self.update_template_button = QPushButton('Update Template', self)
        self.update_template_button.clicked.connect(self.update_template)
        self.update_template_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        template_button_layout.addWidget(self.update_template_button)

        right_layout.addLayout(template_button_layout)

        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 2)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.actions = []
        self.templates = []
        self.current_action_index = None
        self.current_template_id = None
        self.load_templates()

    def add_action(self):
        action_name = self.action_name_input.text()
        action_type = self.type_input.currentText()
        type_target = self.type_target_input.text()
        media = self.media_input.text()
        content = self.content_input.text()
        step = len(self.actions) + 1

        action = {
            'Name': action_name,
            'Type': action_type,
            'Type_target': type_target,
            'Media': media,
            'Step': step,
            'Content': content
        }

        self.actions.append(action)
        self.update_action_list()
        self.clear_form()

    def update_action_list(self):
        self.actions.sort(key=lambda x: x['Step'])
        self.action_list.clear()
        for index, action in enumerate(self.actions):
            action['Step'] = index + 1
            item_text = f"{action['Step']}: {action['Name']}"
            list_item = QListWidgetItem(item_text)
            item_widget = QWidget()
            item_layout = QHBoxLayout()
            item_label = QLabel(item_text)
            delete_button = QPushButton('Delete')
            delete_button.setFixedWidth(50)
            delete_button.clicked.connect(lambda _, idx=index: self.confirm_delete_action(idx))
            item_layout.addWidget(item_label)
            item_layout.addWidget(delete_button)
            item_layout.setStretch(0, 1)
            item_widget.setLayout(item_layout)
            list_item.setSizeHint(item_widget.sizeHint())
            self.action_list.addItem(list_item)
            self.action_list.setItemWidget(list_item, item_widget)

    def edit_action(self, item):
        index = self.action_list.row(item)
        action = self.actions[index]
        self.action_name_input.setText(action['Name'])
        self.type_input.setCurrentText(action['Type'])
        self.type_target_input.setText(action['Type_target'])
        self.media_input.setText(action['Media'])
        self.content_input.setText(action['Content'])
        self.current_action_index = index
        self.add_button.setText("Add as New Action")
        self.update_button.setVisible(True)

    def update_action(self):
        if self.current_action_index is not None:
            action = self.actions[self.current_action_index]
            action['Name'] = self.action_name_input.text()
            action['Type'] = self.type_input.currentText()
            action['Type_target'] = self.type_target_input.text()
            action['Media'] = self.media_input.text()
            action['Content'] = self.content_input.text()
            self.update_action_list()
            if self.current_template_id:
                self.db_manager.update_template(self.current_template_id, self.template_name_input.text(), self.actions)

    def clear_form(self):
        self.action_name_input.clear()
        self.type_input.setCurrentIndex(0)
        self.type_target_input.clear()
        self.media_input.clear()
        self.content_input.clear()
        self.add_button.setText("Add Action")
        self.update_button.setVisible(False)
        self.current_action_index = None

    def confirm_delete_action(self, index):
        reply = QMessageBox.question(self, 'Confirm Delete', 'Are you sure you want to delete this action?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.delete_action(index)

    def delete_action(self, index):
        self.actions.pop(index)
        self.update_action_list()
        if self.current_template_id:
            self.db_manager.update_template(self.current_template_id, self.template_name_input.text(), self.actions)

    def save_template(self):
        file_name = self.template_name_input.text()
        if not file_name:
            QMessageBox.warning(self, "Error", "Please enter a template file name.")
            return

        try:
            self.db_manager.save_template(file_name, self.actions)
            QMessageBox.information(self, "Success", f"Template '{file_name}' saved to database.")
            self.load_templates()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save template: {e}")

    def update_template(self):
        if self.current_template_id is not None:
            try:
                self.db_manager.update_template(self.current_template_id, self.template_name_input.text(), self.actions)
                QMessageBox.information(self, "Success", f"Template updated successfully.")
                self.load_templates()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not update template: {e}")

    def load_templates(self):
        self.templates = self.db_manager.load_templates()
        self.template_list.clear()
        for template in self.templates:
            item_widget = QWidget()
            item_layout = QHBoxLayout()
            item_label = QLabel(template['name'])
            delete_button = QPushButton('Delete')
            delete_button.setFixedWidth(50)
            delete_button.clicked.connect(lambda _, t=template: self.delete_template(t['id']))
            item_layout.addWidget(item_label)
            item_layout.addWidget(delete_button)
            item_layout.setStretch(0, 1)
            item_widget.setLayout(item_layout)
            list_item = QListWidgetItem()
            list_item.setSizeHint(item_widget.sizeHint())
            self.template_list.addItem(list_item)
            self.template_list.setItemWidget(list_item, item_widget)

    def delete_template(self, template_id):
        try:
            self.db_manager.delete_template(template_id)
            QMessageBox.information(self, "Success", "Template deleted from database.")
            self.load_templates()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not delete template: {e}")

    def load_template(self, item):
        item_widget = self.template_list.itemWidget(item)
        template_name = item_widget.findChild(QLabel).text()
        for template in self.templates:
            if template['name'] == template_name:
                self.actions = template['actions']
                self.current_template_id = template['id']
                self.template_name_input.setText(template['name'])
                self.update_action_list()
                break

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
