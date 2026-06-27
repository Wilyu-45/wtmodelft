"""
txt 分割工具 GUI
用户手动添加每段的行数范围，支持自定义文件名
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path


class SplitTxtGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("txt 分割工具")
        self.window.geometry("800x550")

        self.input_dir = Path("txtprocess")
        self.output_dir = Path("input")
        self.selected_file = None
        self.total_lines = 0
        self.ranges = []

        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.window, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = ttk.Label(main_frame, text="txt 分割工具", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 10))

        file_frame = ttk.LabelFrame(main_frame, text="选择文件", padding="8")
        file_frame.pack(fill=tk.X, pady=(0, 5))

        self.file_listbox = tk.Listbox(file_frame, height=4, font=("Arial", 11))
        self.file_listbox.pack(fill=tk.X, pady=(0, 5))
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select)

        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill=tk.X)

        ttk.Button(btn_frame, text="刷新", command=self.load_files).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="选择目录", command=self.select_input_dir).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="批量导出audiotemp.list", command=self.batch_export_from_list).pack(side=tk.LEFT, padx=(20, 0))
        ttk.Button(btn_frame, text="处理input纯文本", command=self.process_input_files).pack(side=tk.LEFT, padx=(20, 0))

        self.dir_label = ttk.Label(main_frame, text=f"文件目录: {self.input_dir}", foreground="gray")
        self.dir_label.pack(anchor=tk.W, pady=(2, 0))

        self.selected_label = ttk.Label(main_frame, text="未选择文件", foreground="red", font=("Arial", 11, "bold"))
        self.selected_label.pack(anchor=tk.W, pady=(5, 5))

        range_frame = ttk.LabelFrame(main_frame, text="添加行范围", padding="8")
        range_frame.pack(fill=tk.X, pady=(0, 5))

        input_row = ttk.Frame(range_frame)
        input_row.pack(fill=tk.X)

        ttk.Label(input_row, text="起始行:").pack(side=tk.LEFT)
        self.start_entry = ttk.Entry(input_row, width=10)
        self.start_entry.pack(side=tk.LEFT, padx=(5, 10))

        ttk.Label(input_row, text="结束行:").pack(side=tk.LEFT)
        self.end_entry = ttk.Entry(input_row, width=10)
        self.end_entry.pack(side=tk.LEFT, padx=(5, 10))

        ttk.Label(input_row, text="文件名:").pack(side=tk.LEFT)
        self.name_entry = ttk.Entry(input_row, width=20)
        self.name_entry.pack(side=tk.LEFT, padx=(5, 10))

        ttk.Button(input_row, text="添加", command=self.add_range).pack(side=tk.LEFT, padx=(10, 5))
        ttk.Button(input_row, text="删除选中", command=self.delete_range).pack(side=tk.LEFT, padx=(5, 5))
        ttk.Button(input_row, text="清空", command=self.clear_ranges).pack(side=tk.LEFT)

        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        ttk.Label(list_frame, text="已添加的范围:").pack(anchor=tk.W)

        columns = ("范围", "文件名")
        self.range_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=8)

        self.range_tree.heading("范围", text="行范围")
        self.range_tree.heading("文件名", text="输出文件名")
        self.range_tree.column("范围", width=200)
        self.range_tree.column("文件名", width=350)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.range_tree.yview)
        self.range_tree.configure(yscrollcommand=scrollbar.set)

        self.range_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.range_tree.bind('<Double-Button-1>', self.edit_name)

        self.status_label = ttk.Label(main_frame, text="请选择文件", foreground="blue")
        self.status_label.pack(anchor=tk.W, pady=(5, 0))

        self.split_btn = ttk.Button(main_frame, text="开始分割", command=self.start_split, state='disabled')
        self.split_btn.pack(fill=tk.X, pady=(5, 0))

        self.load_files()

    def load_files(self):
        self.file_listbox.delete(0, tk.END)

        if not self.input_dir.exists():
            self.file_listbox.insert(tk.END, f"目录不存在: {self.input_dir}")
            return

        txt_files = sorted(self.input_dir.glob("*.txt"))
        list_files = sorted(self.input_dir.glob("*.list"))
        all_files = txt_files + list_files

        if not all_files:
            self.file_listbox.insert(tk.END, "(目录中没有 txt 或 list 文件)")
            return

        for f in all_files:
            self.file_listbox.insert(tk.END, f.name)

        self.status_label.config(text=f"找到 {len(txt_files)} 个 txt 文件, {len(list_files)} 个 list 文件", foreground="green")

    def select_input_dir(self):
        folder = filedialog.askdirectory(initialdir=str(self.input_dir))
        if folder:
            self.input_dir = Path(folder)
            self.dir_label.config(text=f"文件目录: {self.input_dir}")
            self.load_files()

    def on_file_select(self, event):
        selection = self.file_listbox.curselection()
        if selection:
            filename = self.file_listbox.get(selection[0])
            self.selected_file = self.input_dir / filename

            try:
                with open(self.selected_file, 'r', encoding='utf-8') as f:
                    self.total_lines = len(f.readlines())

                self.start_entry.delete(0, tk.END)
                self.start_entry.insert(0, "1")
                self.end_entry.delete(0, tk.END)
                self.end_entry.insert(0, str(self.total_lines))
                self.name_entry.delete(0, tk.END)

                self.selected_label.config(
                    text=f"已选择: {filename} (共 {self.total_lines} 行)",
                    foreground="green"
                )
                self.clear_ranges()
                self.status_label.config(text=f"请添加行范围和文件名，然后点击下方按钮分割", foreground="blue")
                self.start_entry.focus()

            except Exception as e:
                self.status_label.config(text=f"读取文件失败: {e}", foreground="red")

    def add_range(self):
        if not self.selected_file:
            messagebox.showwarning("警告", "请先选择一个文件")
            return

        try:
            start = int(self.start_entry.get().strip())
            end = int(self.end_entry.get().strip())
            name = self.name_entry.get().strip()

            if start < 1 or end > self.total_lines or start > end:
                messagebox.showwarning("警告", f"范围无效，有效范围: 1-{self.total_lines}")
                return

            if not name:
                name = f"{self.selected_file.stem}_part{len(self.ranges)+1:02d}"

            if not name.endswith('.txt'):
                name = name + '.txt'

            self.ranges.append((start, end, name))
            self.ranges.sort(key=lambda x: x[0])
            self.update_range_display()

            self.start_entry.delete(0, tk.END)
            self.end_entry.delete(0, tk.END)
            self.name_entry.delete(0, tk.END)
            self.start_entry.focus()

            self.split_btn.config(state='normal')
            self.status_label.config(
                text=f"已添加 {len(self.ranges)} 个范围，点击下方按钮开始分割",
                foreground="green"
            )

        except ValueError:
            messagebox.showwarning("警告", "请输入有效的数字")

    def edit_name(self, event):
        region = self.range_tree.identify("region", event.x, event.y)
        if region == "cell":
            row_id = self.range_tree.identify_row(event.y)
            column = self.range_tree.identify_column(event.x)

            if column == "#2":
                col_index = 1
                x = self.range_tree.winfo_x() + event.x
                y = self.range_tree.winfo_y() + event.y

                item = self.range_tree.item(row_id)
                current_name = item['values'][1]

                entry_popup = tk.Toplevel(self.window)
                entry_popup.wm_overrideredirect(True)
                entry_popup.wm_geometry(f"+{x}+{y}")

                entry_var = tk.StringVar(value=current_name)
                entry = ttk.Entry(entry_popup, width=30, textvariable=entry_var)
                entry.pack()
                entry.select_range(0, tk.END)
                entry.focus()

                def save_edit(event):
                    new_name = entry_var.get().strip()
                    if new_name:
                        if not new_name.endswith('.txt'):
                            new_name = new_name + '.txt'
                        idx = self.range_tree.index(row_id)
                        start, end, _ = self.ranges[idx]
                        self.ranges[idx] = (start, end, new_name)
                        self.update_range_display()
                    entry_popup.destroy()

                def close_popup(event):
                    entry_popup.destroy()

                entry.bind('<Return>', save_edit)
                entry.bind('<Escape>', close_popup)
                entry.bind('<FocusOut>', close_popup)

                entry_popup.grab_set()

    def delete_range(self):
        selection = self.range_tree.selection()
        if selection:
            idx = self.range_tree.index(selection[0])
            removed = self.ranges.pop(idx)
            self.update_range_display()
            self.status_label.config(
                text=f"已删除: {removed[0]}-{removed[1]} ({removed[2]})",
                foreground="gray"
            )
            if not self.ranges:
                self.split_btn.config(state='disabled')
        else:
            messagebox.showinfo("提示", "请先选中要删除的范围")

    def clear_ranges(self):
        self.ranges = []
        for item in self.range_tree.get_children():
            self.range_tree.delete(item)
        self.split_btn.config(state='disabled')
        if self.selected_file:
            self.selected_label.config(
                text=f"已选择: {self.selected_file.name} (共 {self.total_lines} 行)",
                foreground="green"
            )

    def update_range_display(self):
        for item in self.range_tree.get_children():
            self.range_tree.delete(item)

        for i, (start, end, name) in enumerate(self.ranges, 1):
            self.range_tree.insert('', tk.END, values=(f"行 {start} - {end}", name))

    def start_split(self):
        if not self.ranges:
            messagebox.showwarning("警告", "请先添加至少一个范围")
            return

        try:
            with open(self.selected_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            self.output_dir.mkdir(parents=True, exist_ok=True)

            for i, (start, end, name) in enumerate(self.ranges, 1):
                segment_lines = lines[start-1:end]
                segment_text = ''.join(segment_lines)

                output_path = self.output_dir / name

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(segment_text)

            messagebox.showinfo("完成", f"成功生成 {len(self.ranges)} 个文件\n\n输出目录: {self.output_dir}")
            self.status_label.config(text=f"已完成分割，输出到 {self.output_dir}", foreground="green")

        except Exception as e:
            messagebox.showerror("错误", f"分割失败: {e}")

    def batch_export_from_list(self):
        if self.selected_file and self.selected_file.suffix == '.list':
            list_file = self.selected_file
        else:
            list_file = Path("audiotemp.list")

        if not list_file.exists():
            messagebox.showwarning("警告", f"文件不存在: {list_file}")
            return

        try:
            with open(list_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            exported_count = 0
            error_count = 0

            self.output_dir.mkdir(parents=True, exist_ok=True)

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('|', 3)
                if len(parts) < 4:
                    error_count += 1
                    continue

                output_path_str, folder, language, text = parts
                original_path = Path(output_path_str)
                output_filename = original_path.stem + ".txt"

                output_path = self.output_dir / output_filename

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)

                exported_count += 1

            if exported_count > 0:
                messagebox.showinfo("完成", f"成功导出 {exported_count} 个文件\n\n输出目录: {self.output_dir}")
                self.status_label.config(text=f"批量导出完成: {exported_count} 个文件", foreground="green")
            else:
                messagebox.showwarning("结果", "没有成功导出的文件")
                if error_count > 0:
                    self.status_label.config(text=f"导出失败: {error_count} 行格式错误", foreground="red")

        except Exception as e:
            messagebox.showerror("错误", f"批量导出失败: {e}")

    def process_input_files(self):
        import re

        input_dir = Path("input")

        if not input_dir.exists():
            messagebox.showwarning("警告", f"目录不存在: {input_dir}")
            return

        txt_files = sorted(input_dir.glob("*.txt"))

        if not txt_files:
            messagebox.showwarning("警告", "目录中没有 txt 文件")
            return

        try:
            total_lines = 0
            for txt_file in txt_files:
                pure_text_lines = []

                with open(txt_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        if line.startswith("Dialogue:"):
                            last_brace = line.rfind('}')
                            if last_brace != -1 and last_brace < len(line) - 1:
                                text_content = line[last_brace + 1:].strip()
                            else:
                                parts = line.split(',', 9)
                                if len(parts) >= 10:
                                    text_content = parts[9].strip()
                                else:
                                    text_content = ""
                        else:
                            text_content = line

                        text_content = re.sub(r'\{[^}]*\}', '', text_content)
                        text_content = text_content.strip()

                        if text_content:
                            pure_text_lines.append(text_content)

                        total_lines += 1

                output_path = input_dir / (txt_file.stem + "_纯文本.txt")

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(pure_text_lines))

            messagebox.showinfo("完成", f"成功处理 {len(txt_files)} 个文件\n\n输出目录: {input_dir}")
            self.status_label.config(text=f"处理完成: {len(txt_files)} 个文件", foreground="green")

        except Exception as e:
            messagebox.showerror("错误", f"处理失败: {e}")

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    app = SplitTxtGUI()
    app.run()